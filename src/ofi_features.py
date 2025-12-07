import numpy as np
import pandas as pd

def add_ofi_features(
    df: pd.DataFrame,
    window_short: int = 5,
    window_long: int = 20,
) -> pd.DataFrame:
    """
    Add basic Order Flow Imbalance (OFI) features to a dataframe.

    Assumes df has at least:
        - buy_volume
        - sell_volume
        - mid_price
    """
    # Work on a copy so we don't accidentally modify the original df
    df = df.copy()

    # 1) Basic OFI per time step: net aggressive volume
    df["ofi"] = df["buy_volume"] - df["sell_volume"]

    # 2) Rolling sums of OFI (short and long windows)
    df[f"ofi_sum_{window_short}"] = df["ofi"].rolling(window_short, min_periods=1).sum()
    df[f"ofi_sum_{window_long}"] = df["ofi"].rolling(window_long, min_periods=1).sum()

    # 3) Total traded volume per step
    df["total_volume"] = df["buy_volume"] + df["sell_volume"]

    # 4) Normalized OFI: OFI divided by total volume
    # Avoid division by zero by temporarily replacing 0 with NaN
    df["ofi_norm"] = df["ofi"] / df["total_volume"].replace(0, np.nan)
    # Where total_volume was 0, ofi_norm will be NaN; set those to 0
    df["ofi_norm"] = df["ofi_norm"].fillna(0.0)

    return df

def add_return_and_labels(
    df: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.0001,
) -> pd.DataFrame:
    """
    Add future return and classification labels to the dataframe.

    horizon:
        how many steps ahead to look (e.g., 10 seconds if freq is 1s)
    threshold:
        minimum return (in absolute value) to consider as up/down.
        Values between -threshold and +threshold are labeled as flat (0).
    """
    df = df.copy()

    # 1) Future mid price: value 'horizon' steps ahead
    df["mid_price_future"] = df["mid_price"].shift(-horizon)

    # 2) Future return: (P_{t+H} - P_t) / P_t
    df["ret_future"] = (df["mid_price_future"] - df["mid_price"]) / df["mid_price"]

    # 3) Classification label based on future return
    def label_from_return(r: float) -> int:
        if r > threshold:
            return 1   # up
        elif r < -threshold:
            return -1  # down
        else:
            return 0   # flat

    df["direction"] = df["ret_future"].apply(label_from_return)

    return df
