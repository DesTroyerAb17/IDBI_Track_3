"""
Data loader abstraction layer.

TODAY: reads synthetic_msme_data.csv (hackathon prototype).
POST-SANDBOX / PRODUCTION: swap the body of these two functions to call
IDBI's sandbox GST/UPI/AA/EPFO/utility APIs instead. Nothing else in the
app (scoring.py, app.py) needs to change — they only call these two
functions and don't care where the data comes from.
"""

import pandas as pd

DATA_PATH = "synthetic_msme_data.csv"


def list_business_ids():
    df = pd.read_csv(DATA_PATH)
    return df["business_id"].tolist()


def get_business_record(business_id: str) -> dict:
    """
    Returns a single business's raw record as a dict, in the schema
    scoring.py expects. In production this would fan out to:
      - GST portal / GSP API      -> turnover, filing regularity
      - AA consent + fetch        -> bank_balance_monthly, bounce_rate
      - EPFO employer API         -> employee/payroll signals
      - Discom / utility API      -> utility_kwh_year etc.
      - Internal CIBIL/Experian pull -> promoter_cibil
    ... and merge them into the same dict shape below.
    """
    df = pd.read_csv(DATA_PATH)
    row = df[df["business_id"] == business_id].iloc[0]
    return row.to_dict()