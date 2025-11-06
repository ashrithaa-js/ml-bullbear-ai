import yfinance as yf
import pandas as pd
import time
import ssl

def get_stock_data(ticker="AAPL", start="2020-01-01", end="2025-01-01", max_retries=3):
    """
    Download stock data with retry logic and SSL error handling.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with stock data
    """
    
    for attempt in range(max_retries):
        try:
            # Try downloading with timeout
            df = yf.download(
                ticker, 
                start=start, 
                end=end, 
                auto_adjust=True, 
                progress=False
            )
            
            # Check if we got valid data
            # Note: yfinance may return empty DataFrame on SSL/connection errors without raising exception
            if df is None or df.empty:
                # Check if this might be a connection issue (yfinance prints errors but doesn't always raise)
                # We'll treat empty data as potentially a connection error and retry
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"⚠️ Download attempt {attempt + 1} returned no data for {ticker}")
                    print(f"   This may be due to connection issues. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise ConnectionError(
                        f"Failed to download data for {ticker} after {max_retries} attempts. "
                        f"No data returned - this may indicate a connection/SSL issue.\n"
                        f"💡 Possible solutions:\n"
                        f"   - Check your internet connection\n"
                        f"   - Try again later (the data provider may be temporarily unavailable)\n"
                        f"   - Check firewall/proxy settings"
                    )
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Drop rows with all NaN values
            df.dropna(inplace=True)
            
            # Final validation
            if df.empty:
                raise ValueError(f"Data is empty after cleaning for {ticker}")
            
            return df
            
        except (ConnectionError, ValueError) as e:
            # If it's already a ConnectionError we raised, don't retry
            if isinstance(e, ConnectionError):
                raise
            # Otherwise re-raise as-is
            raise
        except Exception as e:
            # Check if it's a connection/SSL error
            error_str = str(e).lower()
            is_connection_error = (
                isinstance(e, (ssl.SSLError, ConnectionError, TimeoutError)) or
                "ssl" in error_str or
                "connection" in error_str or
                "timeout" in error_str or
                "recv failure" in error_str or
                "curl" in error_str
            )
            
            if is_connection_error and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"⚠️ Download attempt {attempt + 1} failed: {str(e)}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            elif is_connection_error:
                raise ConnectionError(
                    f"Failed to download data for {ticker} after {max_retries} attempts. "
                    f"SSL/Connection error: {str(e)}\n"
                    f"💡 Possible solutions:\n"
                    f"   - Check your internet connection\n"
                    f"   - Try again later (the data provider may be temporarily unavailable)\n"
                    f"   - Check firewall/proxy settings"
                )
            else:
                # For non-connection errors, don't retry, just raise
                raise ValueError(f"Error downloading data for {ticker}: {str(e)}")
    
    # Should never reach here, but just in case
    raise ValueError(f"Failed to download data for {ticker} after {max_retries} attempts")
