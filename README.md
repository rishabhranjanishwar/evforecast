# 🚗 Electric Vehicle Demand Forecasting (ADS Lab 5)

## 📌 Project Overview

This project forecasts **Electric Vehicle (EV) demand** using historical transaction data, synthetic data generation, and machine learning models. It is part of **Advanced Data Science Lab 5** coursework.

The project helps policymakers, manufacturers, and urban planners understand future EV demand trends (2025-2026) for better infrastructure & policy planning.

---

## ✅ Features

- **Data Preprocessing** – Cleaning and structuring raw EV transaction data.
- **Synthetic Data Generation** – Simulated EV demand for 2023-2024.
- **Modeling Techniques:**
  - Linear Regression
  - Ridge Regression
  - ARIMA (Time-Series Forecasting)
- **Future Forecasting** – Predicting EV demand for 2025-2026.
- **Data Visualization** – Trend analysis for historical, synthetic, and forecasted data.

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Libraries:**
  - pandas, numpy
  - scikit-learn
  - statsmodels
  - matplotlib

---

## 📂 Project Structure

```
ADS-Lab5/
│
├── ADS_Lab5.ipynb          # Jupyter Notebook (main analysis)
├── requirements.txt        # Dependencies
├── README.md               # Project Documentation
└── data/
    └── title_transactions.csv   # Raw EV transaction data
```

---

## ⚙️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/ADS-Lab5.git
cd ADS-Lab5
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Open the Notebook**

```bash
jupyter notebook ADS_Lab5.ipynb
```

---

## ▶️ Usage

1. Run the notebook cell by cell.
2. Ensure `title_transactions.csv` is present inside the `data/` folder.
3. View forecasted EV demand for **2025-2026** via plots & printed predictions.

---

## 📊 Sample Output

- EV demand trends till 2021 (Historical)
- Synthetic EV demand (2023-2024)
- Forecasted EV demand (2025-2026) using **Linear, Ridge, and ARIMA** models

---

## 📜 License

This project is licensed under the **MIT License** – free to use and modify.

---

## 👨‍💻 Author

*Developed as part of ADS Lab coursework.*

