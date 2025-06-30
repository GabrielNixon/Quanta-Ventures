# RL-Based Regime-Aware Portfolio Strategy

This repository contains my submission for the Quanta Ventures Quantitative Research Fellowship. It implements a regime-aware portfolio allocation strategy using a Recurrent PPO (LSTM) reinforcement learning agent trained on daily returns from SPY, LQD, BIL, and Gold.

## Strategy Overview

The agent learns to allocate capital across assets based on historical returns and a regime signal. The reward function is shaped to encourage high Sharpe ratios, penalize excessive turnover, and handle volatility spikes. Regime predictions help the agent shift risk exposure during different market conditions.

Key features:
- Sharpe-normalized reward with transaction penalties
- Regime input from volatility and credit spread signals
- Reward clipping and random shocks for training stability
- Capital reset every 30 steps to simulate reinvestment cycles

## Data

Data is pulled from Yahoo Finance from 2020 to 2025 for the following tickers:
- `SPY` – S&P 500
- `LQD` – Investment Grade Bonds
- `BIL` – T-Bills
- `GC=F` – Gold

The regime flag is based on rolling volatility and bond–T-bill spread.

## Results

The strategy was trained on data from 2020–2024 and tested on 2025.

IN-SAMPLE (2020–2024)
CAGR : 27.6%
Max Drawdown : 8.9%
Calmar Ratio : 3.10
Sharpe Ratio : 1.67

OUT-OF-SAMPLE (2025)
CAGR : 13.4%
Max Drawdown : 7.2%
Calmar Ratio : 1.86
Sharpe Ratio : 1.41


## Notes

The training data mostly covers the COVID and post-COVID period, which isn’t ideal due to high market distortion. Performance may improve with a broader, more balanced dataset.
