"""Calculate technical indicators from a prepared candle CSV.

Processing policy
-----------------
* Every segment is calculated independently. No price change, rolling
  window, or smoothing state crosses a segment boundary.
* Synthetic candles participate in calculations as flat-price,
  zero-volume observations.
* Synthetic and non-tradable candles are never decision-eligible.
* All features use only the current candle and earlier candles.
* Missing or undefined values remain missing; no backfilling is used.
* Periods are measured in candles, not calendar minutes.
* Account state and position management are outside this module.

Calculation conventions
-----------------------
* EMA: SMA seed over N observations, then alpha = 2 / (N + 1).
* Wilder smoothing: SMA seed over N observations, then alpha = 1 / N.
* RSI: Wilder-smoothed close-to-close gains and losses.
  Both averages zero -> 50; only average loss zero -> 100.
* ATR: Wilder-smoothed true range. The first true range in each
  segment is high - low because no prior segment close is used.
* Bollinger Bands: SMA(20), population standard deviation (ddof=0),
  and a multiplier of 2.
* Relative volume: current volume / mean of the previous 20 candles.
* Zero denominators produce missing values.
* Columns ending in "_pct" use percentage units: 1.0 means 1%.
  bb_percent_b and volume_ratio_20 are unscaled ratios.
  rsi14_delta_1 uses RSI points.

Readiness
---------
indicators_ready means all feature columns are finite.
warmup_complete means more than --warmup-bars observations have elapsed
in the current segment.
decision_eligible additionally requires a real, tradable candle.

A decision can use a row only after that candle has closed. Eligibility
does not authorize execution at the same candle's closing price.

Examples:
    uv run python -m s1trader.indicators data/prepared/example.csv
    uv run python -m s1trader.indicators data/prepared/example.csv \
        --warmup-bars 300
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .validate import Report, validate


UTC = timezone.utc
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume", "quote_volume"]
METADATA_COLUMNS = [
    "is_synthetic",
    "can_trade",
    "segment_id",
    "gap_before_minutes",
]

FEATURE_COLUMNS = [
    "return_1_pct",
    "return_5_pct",
    "return_15_pct",
    "sma20",
    "ema20",
    "ema50",
    "close_to_ema20_pct",
    "close_to_ema50_pct",
    "ema20_change_5_pct",
    "rsi14",
    "rsi14_delta_1",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_hist_pct",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "bb_width_pct",
    "atr14",
    "atr14_pct",
    "volume_ratio_20",
]

STATUS_COLUMNS = [
    "segment_bar_number",
    "indicators_ready",
    "warmup_complete",
    "decision_eligible",
    "warmup_bars",
    "indicator_version",
]


def seeded_smoothing(
    values: pd.Series,
    period: int,
    alpha: float,
) -> pd.Series:
    """Apply recursive smoothing initialized with an arithmetic mean.

    Leading missing values are allowed, as needed for RSI changes and
    the MACD signal. Missing values after the first valid observation
    are rejected rather than silently changing the smoothing behavior.

    The first output occurs at the Nth valid observation:
        seed = mean(first N valid observations)
        next = previous + alpha * (current - previous)
    """
    source = values.to_numpy(dtype=float)
    result = np.full(len(source), np.nan)
    valid = np.flatnonzero(np.isfinite(source))

    if not len(valid):
        return pd.Series(result, index=values.index)

    start = int(valid[0])
    if not np.isfinite(source[start:]).all():
        raise ValueError("Smoothing input contains internal missing values.")

    seed_index = start + period - 1
    if seed_index >= len(source):
        return pd.Series(result, index=values.index)

    result[seed_index] = source[start:seed_index + 1].mean()

    for index in range(seed_index + 1, len(source)):
        previous = result[index - 1]
        result[index] = previous + alpha * (source[index] - previous)

    return pd.Series(result, index=values.index)


def ema(values: pd.Series, period: int) -> pd.Series:
    """Calculate an SMA-seeded exponential moving average."""
    return seeded_smoothing(values, period, alpha=2 / (period + 1))


def wilder(values: pd.Series, period: int) -> pd.Series:
    """Calculate Wilder's moving average with an SMA seed."""
    return seeded_smoothing(values, period, alpha=1 / period)


def safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide aligned series, leaving zero-denominator results missing."""
    return numerator / denominator.where(denominator != 0)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Wilder RSI using within-segment price changes.

    The first price change is missing, so RSI(14) needs 15 closes.
    A completely flat seed receives a neutral RSI of 50.
    """
    change = close.diff()
    average_gain = wilder(change.clip(lower=0), period)
    average_loss = wilder(-change.clip(upper=0), period)

    result = 100 * safe_ratio(
        average_gain,
        average_gain + average_loss,
    )
    flat = (average_gain == 0) & (average_loss == 0)
    return result.mask(flat, 50.0)


def calculate_segment(
    frame: pd.DataFrame,
    warmup_bars: int,
) -> pd.DataFrame:
    """Calculate features for one contiguous, equally spaced segment.

    Prepared synthetic candles must already be present. This function
    neither inserts candles nor carries state from another segment.
    """
    result = frame.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]
    volume = result["volume"]

    for period in (1, 5, 15):
        result[f"return_{period}_pct"] = (
            safe_ratio(close, close.shift(period)) - 1
        ) * 100

    result["sma20"] = close.rolling(20, min_periods=20).mean()
    result["ema20"] = ema(close, 20)
    result["ema50"] = ema(close, 50)

    for period in (20, 50):
        result[f"close_to_ema{period}_pct"] = (
            safe_ratio(close, result[f"ema{period}"]) - 1
        ) * 100

    result["ema20_change_5_pct"] = (
        safe_ratio(result["ema20"], result["ema20"].shift(5)) - 1
    ) * 100

    result["rsi14"] = rsi(close, 14)
    result["rsi14_delta_1"] = result["rsi14"].diff()

    result["macd"] = ema(close, 12) - ema(close, 26)
    result["macd_signal"] = ema(result["macd"], 9)
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["macd_hist_pct"] = (
        safe_ratio(result["macd_hist"], close) * 100
    )

    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    result["bb_middle"] = result["sma20"]
    result["bb_upper"] = result["bb_middle"] + 2 * deviation
    result["bb_lower"] = result["bb_middle"] - 2 * deviation

    band_width = result["bb_upper"] - result["bb_lower"]
    result["bb_percent_b"] = safe_ratio(
        close - result["bb_lower"],
        band_width,
    )
    result["bb_width_pct"] = (
        safe_ratio(band_width, result["bb_middle"]) * 100
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    result["atr14"] = wilder(true_range, 14)
    result["atr14_pct"] = safe_ratio(result["atr14"], close) * 100

    # Exclude the current candle from its own reference average.
    previous_volume_mean = volume.shift(1).rolling(
        20, min_periods=20
    ).mean()
    result["volume_ratio_20"] = safe_ratio(
        volume,
        previous_volume_mean,
    )

    # Preserve undefined values as missing, including numeric overflow.
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    result["segment_bar_number"] = np.arange(1, len(result) + 1)
    result["indicators_ready"] = result[FEATURE_COLUMNS].notna().all(axis=1)
    result["warmup_complete"] = (
        result["segment_bar_number"] > warmup_bars
    )
    result["decision_eligible"] = (
        result["indicators_ready"]
        & result["warmup_complete"]
        & result["can_trade"]
        & ~result["is_synthetic"]
    )

    # Store the configurable policy alongside a version for fixed formulas.
    result["warmup_bars"] = warmup_bars
    result["indicator_version"] = "1.0"
    return result


def load_prepared(path: Path) -> pd.DataFrame:
    """Validate candles and verify preparation metadata before calculation.

    The raw validator's expected warnings about zero volumes and long
    gaps are not printed individually. Metadata checks ensure that every
    remaining gap starts a new segment and every synthetic candle follows
    the documented flat-price, zero-volume policy.
    """
    report = Report()
    validate(path, report)

    if report.errors:
        report.display()
        raise ValueError(
            f"Input validation found {report.errors} error(s)."
        )

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(METADATA_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            "Missing preparation columns: " + ", ".join(sorted(missing))
        )

    collisions = set(FEATURE_COLUMNS + STATUS_COLUMNS) & set(frame.columns)
    if collisions:
        raise ValueError("Input already contains indicator output columns.")

    for name in NUMERIC_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} cannot be represented as finite numbers.")

    for name in ("is_synthetic", "can_trade"):
        normalized = frame[name].str.strip().str.lower()
        if not normalized.isin(["true", "false"]).all():
            raise ValueError(f"{name} must contain only true or false.")
        frame[name] = normalized.eq("true")

    for name in ("segment_id", "gap_before_minutes"):
        if not frame[name].str.fullmatch(r"\d+").all():
            raise ValueError(f"{name} must contain nonnegative integers.")
        frame[name] = frame[name].astype("int64")

    minutes = int(frame["interval_minutes"].iloc[0])
    timestamps = pd.to_datetime(frame["time_utc"], utc=True)
    elapsed = timestamps.diff().dt.total_seconds().div(60)
    actual_gap = (elapsed - minutes).fillna(0)

    if not actual_gap.eq(frame["gap_before_minutes"]).all():
        raise ValueError("gap_before_minutes disagrees with timestamps.")

    # Each preserved gap increments the segment ID exactly once.
    expected_segments = actual_gap.gt(0).cumsum()
    if not frame["segment_id"].eq(expected_segments).all():
        raise ValueError("segment_id disagrees with preserved gaps.")

    synthetic = frame["is_synthetic"]
    starts_segment = frame["segment_id"].ne(frame["segment_id"].shift())
    if (synthetic & starts_segment).any():
        raise ValueError("A segment cannot start with a synthetic candle.")

    for name in PRICE_COLUMNS:
        if (
            synthetic
            & frame[name].ne(frame["close"].shift())
        ).any():
            raise ValueError(
                "Synthetic OHLC values must equal the previous close."
            )

    if (
        synthetic
        & (frame["volume"].ne(0) | frame["quote_volume"].ne(0))
    ).any():
        raise ValueError("Synthetic candles must have zero volumes.")

    expected_can_trade = (
        ~synthetic
        & frame["volume"].gt(0)
        & frame["quote_volume"].gt(0)
    )
    if not frame["can_trade"].eq(expected_can_trade).all():
        raise ValueError("can_trade disagrees with preparation policy.")

    return frame


def build_features(
    source: Path,
    output_dir: Path,
    warmup_bars: int,
) -> Path:
    """Build an enriched CSV without modifying the prepared input.

    Output is written to a temporary sibling file and renamed after a
    successful write. Missing indicator values are serialized as empty
    CSV cells. Booleans use lowercase true/false strings.
    """
    frame = load_prepared(source)
    segments = [
        calculate_segment(group, warmup_bars)
        for _, group in frame.groupby("segment_id", sort=False)
    ]
    result = pd.concat(segments, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"{source.stem}_features_{stamp}.csv"
    temporary = output.with_suffix(".csv.part")

    serialized = result.copy()
    boolean_columns = [
        "is_synthetic",
        "can_trade",
        "indicators_ready",
        "warmup_complete",
        "decision_eligible",
    ]
    for name in boolean_columns:
        serialized[name] = serialized[name].map(
            {True: "true", False: "false"}
        )

    with temporary.open("x", encoding="utf-8", newline="") as file:
        serialized.to_csv(file, index=False, na_rep="")
    temporary.replace(output)

    print("\nIndicator calculation complete")
    print(f"Rows:                {len(result):,}")
    print(f"Segments:            {result['segment_id'].nunique():,}")
    print(f"Feature columns:     {len(FEATURE_COLUMNS):,}")
    print(f"Warmup bars/segment: {warmup_bars:,}")
    print(f"Indicators ready:    {result['indicators_ready'].sum():,}")
    print(f"Decision eligible:   {result['decision_eligible'].sum():,}")
    print(f"Saved: {output.resolve()}")
    return output


def main() -> None:
    """Build indicator features from one prepared candle CSV."""
    parser = argparse.ArgumentParser(
        description="Calculate documented, segment-aware candle indicators."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=200,
        help="Initial candles excluded from decisions per segment (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
    )
    args = parser.parse_args()

    if args.warmup_bars < 0:
        parser.error("--warmup-bars must be nonnegative.")

    try:
        build_features(
            args.csv_path,
            args.output_dir,
            args.warmup_bars,
        )
    except (
        OSError,
        UnicodeError,
        csv.Error,
        pd.errors.ParserError,
        ValueError,
        OverflowError,
    ) as error:
        parser.exit(1, f"Indicator calculation failed: {error}\n")


if __name__ == "__main__":
    main()