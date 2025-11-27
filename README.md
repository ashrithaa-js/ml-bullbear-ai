# Algorithmic Trading - Machine Learning Project 

## Overview
This project focuses on developing machine learning models to predict next-day stock price movements using historical financial data. The models leverage advanced algorithms and interpretability frameworks to provide transparent and trustworthy trading signals.

## Algorithms Used
- **Random Forest Regressor:** An ensemble method combining multiple decision trees to improve prediction accuracy and robustness, especially with noisy, high-dimensional financial data.
- **Gradient Boosting:** A sequential ensemble approach that iteratively corrects model errors, effective for capturing subtle patterns in time series.
- **ARIMA (AutoRegressive Integrated Moving Average):** A classical statistical model for capturing linear trends and seasonality in financial time series.
- **Reinforcement Learning:** An agent-based approach that learns trading strategies from interaction with the market environment through rewards.

## Dataset
- Source: Yahoo Finance via the yfinance Python library
- Includes historical stock data (Open, High, Low, Close, Volume)
- Derived technical indicators: Moving Averages, Relative Strength Index (RSI), MACD, Volatility

## Key Features
- Predicts next-day price movement (Up/Down)
- Uses SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to explain model predictions
- Enhances decision transparency by providing human-understandable insights into why a certain prediction was made
- Supports explainability focused trading to improve investors’ trust and comprehension of automated models

## Tech Stack
- **Language:** Python  
- **Libraries:**  
  - yfinance (data collection)  
  - pandas, numpy (data handling)  
  - scikit-learn (Random Forest, Gradient Boosting)  
  - statsmodels (ARIMA)  
  - shap, lime (model explainability)  
  - matplotlib, seaborn (visualizations)  
  - gym, stable-baselines3 (reinforcement learning simulation)

## Conclusion
This project combines multiple machine learning techniques with explainability frameworks to deliver reliable, interpretable predictions for stock price movements. It provides a structured foundation for building transparent and effective algorithmic trading systems.

---
