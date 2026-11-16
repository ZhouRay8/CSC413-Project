# CSC413-Project, Does Point History Improve Tennis Point Prediction?
**A Comparison of Sequence and Count Models**

**CSC413 Course Project — University of Toronto Mississauga**
* **Authors:** Sam Shojairahnama, Haozhe Huo, Jingcheng Liang, Raymond Zhou

---

## Overview
Classical tennis forecasting models rely on the assumption that points are independent and identically distributed (i.i.d.) with fixed serve-win probabilities. This project investigates whether sequential point history improves next-point probability predictions (made prior to the first serve) beyond current score states, prematch player statistics, and order-free in-match point counts.

We evaluate an **LSTM residual correction architecture**—which adds a next-server-dependent recurrent adjustment to a frozen count-based MLP—against current-state baselines (Logistic Regression, MLP) and order-free count controls across **32,115 ATP and WTA singles matches** (4.76 million points, 2012–2017).

---

## Repository Structure
```text
CSC413-Project/
├── data/
│   ├── raw/                  # Raw TennisVisuals ATP/WTA point-by-point CSVs
│   └── processed/            # Cleaned & decoded parquet files (generated)
├── src/
│   └── clean_data.py         # Data deduplication, retirement filtering, server verification & decoding
├── README.md
└── requirements.txt
