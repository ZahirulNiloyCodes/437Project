# 437Project
# Used Car Resale Price Prediction (CSE437)

## Problem Statement
Secondary vehicle markets suffer from high pricing volatility and asymmetric information. This project develops a machine learning regression pipeline to predict fair market vehicle values from specifications, wear metrics, and geographic attributes.

## Dataset
- **Source:** Kaggle (Austin Reese Craigslist Cars/Trucks dataset)
- **Size:** ~426,000 listings, 26 features (~1.4 GB raw)
- **Target Variable:** `price` (evaluated as continuous on a logarithmic scale)

## Research Questions
1. How non-linearly does vehicle age vs. mileage (`odometer`) impact valuation across different vehicle categories (e.g., trucks vs. sedans)?
2. Controlling for vehicle age, condition, and manufacturer, does listing region/state create statistically significant price divergence?
3. Which factor causes a steeper price penalty: severe cosmetic/mechanical condition downgrades or a compromised title status (e.g., *salvage* vs. *clean*)?

## How to Run
1. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
