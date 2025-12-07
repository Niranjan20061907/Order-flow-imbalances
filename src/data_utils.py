import numpy as np
import pandas as pd


def generate_synthetic_lob_trades(
    n_steps: int = 10_000,
    dt_seconds: float = 1.0,
    seed: int = 42,
):
    """
    Generate synthetic limit order book (LOB) and trades data.

    Returns:
        lob_df: DataFrame with columns [timestamp, bid_price, bid_size, ask_price, ask_size]
        trades_df: DataFrame with columns [timestamp, price, size, side]
                   side = 1 for buy, -1 for sell
    """
    # random number generator
    rng = np.random.default_rng(seed)

    # 1) create a sequence of timestamps
    timestamps = pd.date_range(
        start="2025-01-01 09:30:30",
        periods=n_steps,
        freq=pd.to_timedelta(dt_seconds, unit="s"),
    )

    # 2) simulate a simple random walk for mid price
    mid0 = 100.0      # starting price
    vol = 0.01        # volatility per step (standard deviation of price change)
    shocks = rng.normal(0, vol, size=n_steps)  # random changes each step
    mid_prices = mid0 + np.cumsum(shocks)      # cumulative sum = random walk

    # 3) create bid/ask prices around the mid price
    spread = 0.02  # constant spread
    bid_prices = mid_prices - spread / 2
    ask_prices = mid_prices + spread / 2

    # 4) random sizes at bid and ask
    bid_sizes = rng.integers(10, 100, size=n_steps)
    ask_sizes = rng.integers(10, 100, size=n_steps)

    # build the LOB dataframe
    lob_df = pd.DataFrame(
        {
            "timestamp": timestamps,   # NOTE: singular name "timestamp"
            "bid_price": bid_prices,
            "bid_size": bid_sizes,
            "ask_price": ask_prices,
            "ask_size": ask_sizes,
        }
    )

    # 5) simulate trades: one per time step
    sides = rng.choice([1, -1], size=n_steps)  # 1 = buy, -1 = sell

    # trade occurs at best ask if buy, at best bid if sell
    trade_prices = np.where(sides == 1, ask_prices, bid_prices)
    trade_sizes = rng.integers(1, 50, size=n_steps)

    trades_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": trade_prices,
            "size": trade_sizes,
            "side": sides,
        }
    )

    return lob_df, trades_df


def make_resampled_dataframe(
    lob_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    freq: str = "1S",
) -> pd.DataFrame:
    """
    Combine LOB + trades into a single time series dataframe.

    Output columns:
        timestamp, bid_price, ask_price, bid_size, ask_size,
        mid_price, buy_volume, sell_volume
    """
    # 1) copy to avoid modifying originals
    lob = lob_df.copy()
    trades = trades_df.copy()

    # ensure timestamps are datetime and sorted
    lob["timestamp"] = pd.to_datetime(lob["timestamp"])
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])

    lob = lob.set_index("timestamp").sort_index()
    trades = trades.set_index("timestamp").sort_index()

    # 2) resample LOB: last known state in each time bucket
    lob_resampled = lob.resample(freq).last().ffill()

    # 3) separate buy and sell trades
    buy_trades = trades[trades["side"] == 1]
    sell_trades = trades[trades["side"] == -1]

    # sum sizes within each time bucket
    buy_vol = buy_trades["size"].resample(freq).sum().rename("buy_volume")
    sell_vol = sell_trades["size"].resample(freq).sum().rename("sell_volume")

    # 4) join everything into one dataframe
    df = lob_resampled.join([buy_vol, sell_vol], how="left").fillna(0.0)

    # 5) compute mid price
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2

    # 6) reset index to have timestamp as a column again
    df = df.reset_index().rename(columns={"index": "timestamp"})

    return df