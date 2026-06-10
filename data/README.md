# Data

## Included: 500-row sample

`sample_500.csv` — first 500 rows of the IBM Telco Churn dataset, committed for tests.
The full dataset (7,043 rows) is excluded from the repo (see `.gitignore`).

## Full dataset (recommended for training)

Download `Telco-Customer-Churn.csv` from the IBM Cognos mirror (no login required):

```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
     -o data/telco_churn.csv
```

Source: IBM Corporation (Telco Customer Churn sample dataset, IBM Cognos Analytics).
The pipeline (`src/churn/data.py`) loads the full file if present, otherwise the sample.
