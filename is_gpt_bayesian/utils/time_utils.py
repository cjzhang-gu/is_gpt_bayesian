import pandas as pd

def get_date() -> str:
    return pd.Timestamp.now(tz='US/Eastern').strftime('%Y%m%d_%Z%z')

def get_secondstamp() -> str:
    return pd.Timestamp.now(tz='US/Eastern').strftime('%Y%m%d_%H%M%S_%Z%z')

def get_microsecondstamp() -> str:
    return pd.Timestamp.now(tz='US/Eastern').strftime('%Y%m%d_%H%M%S_%f_%Z%z')