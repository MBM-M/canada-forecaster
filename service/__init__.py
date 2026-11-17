"""Refresh service for the Canada Economic Forecaster.

Watches (simulated) StatCan releases, refits the forecaster's models, promotes a new model only if it beats the
current one on a rolling-origin backtest, and publishes one nowcast and one brief per release.
"""
