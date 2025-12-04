# 📊 Order Flow Imbalance (OFI) – Quant Research Project

This project explores whether **order flow imbalance (OFI)** can predict **short-term price movements**.

I build a pipeline from **raw order book / trade data → OFI features → ML models → (optional) trading signals**.  
This is a learning / research project for quantitative finance and market microstructure.

---

## 🧠 What is Order Flow Imbalance?

**Order flow imbalance (OFI)** tries to measure the *net buying vs selling pressure* over a short period.

Intuition:

- If more **aggressive buy volume** hits the ask than sell volume hits the bid ⇒ upward pressure on price
- If more **aggressive sell volume** hits the bid ⇒ downward pressure

A simple definition over a small window Δt:

\[
\text{OFI}(t, \Delta t) = \sum_k \text{signed\_volume}_k
\]

where each event \(k\) contributes:

- **+volume** if it represents buy pressure (market buy, bid added, ask cancelled)
- **−volume** if it represents sell pressure (market sell, ask added, bid cancelled)

In this project, I start with a simpler approximation based on **buy vs sell volume** in each time bucket and then extend to richer definitions.

---

## 🎯 Project Goals

- Clean and align **order book + trades** data
- Compute **OFI features** over different time windows
- Create **labels**:
  - Regression: future returns
  - Classification: up / down / flat price direction
- Train and compare:
  - ✅ Baseline models: Linear / Logistic Regression
  - ✅ Basic Neural Network (MLP)
- (Optional) Build a tiny **backtest** using model predictions

---

## 📁 Repository Structure

```text
order-flow-imbalance/
│
├── data/
│   ├── raw/                # Raw order book and trades data (not committed to Git ideally)
│   └── processed/          # Cleaned & merged data, OFI features, labels
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_ofi_features_and_labels.ipynb
│   └── 03_models_baseline_vs_nn.ipynb
│
├── src/
│   ├── data_utils.py       # Functions for loading, cleaning, resampling data
│   ├── ofi_features.py     # Functions to compute OFI and related features
│   └── models.py           # Model training helpers (baseline + NN)
│
├── README.md               # Project documentation (this file)
└── requirements.txt        # Python dependencies
