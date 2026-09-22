# src/s1trader/collect.py
"""Download completed Upbit minute candles and save them as CSV.

The collection window is [start, end), where end is the start of the
current candle. This excludes the candle that is still forming.

Candle timestamps identify the beginning of each interval and are stored
in UTC. Results are sorted chronologically and deduplicated by timestamp.
Missing candles are reported, but never synthesized.

All results are held in memory until collection completes. Interrupted
downloads cannot be resumed.

Examples:
    uv run python -m s1trader.collect --period 90d
    uv run python -m s1trader.collect --period 20d --interval 3m
    uv run python -m s1trader.collect --market KRW-ETH --interval 1h
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx


UTC = timezone.utc
BASE_URL = "https://api.upbit.com/v1/candles/minutes"
SUPPORTED_MINUTES = {1, 3, 5, 10, 15, 30, 60, 240}

# OHLC prices and quote_volume use the quote currency.
# For KRW-BTC, prices and quote_volume are in KRW; volume is in BTC.
FIELDS = [
    "market",
    "interval_minutes",
    "time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]

# Pace sequential requests below the exchange's nominal limit.
# Other clients sharing the same public IP can consume the same quota.
REQUEST_INTERVAL = 0.15
MAX_ATTEMPTS = 5


def parse_duration(value: str) -> timedelta:
    """Parse a positive duration for an argparse option.

    Supported suffixes are m (minutes), h (hours), d (days), and w (weeks).
    Days and weeks represent fixed durations of 24 hours and 7 days.

    Raises:
        argparse.ArgumentTypeError: If the format is invalid.
    """
    match = re.fullmatch(r"([1-9]\d*)([mhdw])", value.lower())
    if not match:
        raise argparse.ArgumentTypeError(
            "Use a positive integer followed by m, h, d, or w: 12h, 90d."
        )
    amount = int(match[1])
    seconds = {"m": 60, "h": 3600, "d": 86400, "w": 604800}[match[2]]
    return timedelta(seconds=amount * seconds)


def parse_interval(value: str) -> int:
    """Convert a candle interval to an Upbit-supported minute count.

    Equivalent forms such as 60m and 1h are accepted.

    Raises:
        argparse.ArgumentTypeError: If the interval is unsupported.
    """
    match = re.fullmatch(r"([1-9]\d*)(m|h)", value.lower())
    minutes = (
        int(match[1]) * (60 if match[2] == "h" else 1)
        if match else 0
    )
    if minutes not in SUPPORTED_MINUTES:
        raise argparse.ArgumentTypeError(
            "Supported intervals: 1m, 3m, 5m, 10m, 15m, 30m, 1h, 4h."
        )
    return minutes


def iso_utc(value: datetime) -> str:
    """Format an already UTC-aware datetime with second precision.

    This helper formats timestamps; it does not convert other timezones
    to UTC.
    """
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def candle_time(candle: dict) -> datetime:
    """Return the candle's opening timestamp as a UTC-aware datetime."""
    # Upbit's UTC field has no explicit timezone suffix.
    return datetime.fromisoformat(
        candle["candle_date_time_utc"]
    ).replace(tzinfo=UTC)


def request_page(
    client: httpx.Client,
    market: str,
    minutes: int,
    cursor: datetime,
) -> list[dict]:
    """Fetch up to 200 candles preceding the exclusive UTC cursor.

    Transport errors, HTTP 429, and server errors are retried with
    exponential backoff. A numeric Retry-After header can extend the wait.
    HTTP 418 stops collection immediately rather than retrying a block.

    The caller owns the HTTP client and its connection lifecycle.

    Raises:
        httpx.HTTPError: If retries are exhausted or another HTTP error occurs.
        RuntimeError: If blocked or the response is not a candle list.
        ValueError: If the response cannot be decoded as JSON.
    """
    params = {
        "market": market,
        "count": 200,
        "to": iso_utc(cursor),
    }

    for attempt in range(MAX_ATTEMPTS):
        # Apply pacing to retries as well as initial requests.
        time.sleep(REQUEST_INTERVAL)

        try:
            response = client.get(
                f"{BASE_URL}/{minutes}",
                params=params,
            )
        except httpx.TransportError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 418:
            raise RuntimeError(
                "Upbit temporarily blocked requests (HTTP 418). "
                "Stop and check the response: " + response.text
            )

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_ATTEMPTS - 1:
                response.raise_for_status()

            # Only numeric Retry-After values are handled here.
            # Otherwise, fall back to exponential backoff.
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 0
            time.sleep(max(delay, 2 ** attempt))
            continue

        response.raise_for_status()

        # Leave a short pause when the shared per-second quota is nearly
        # exhausted. The legacy "min" field is intentionally ignored.
        remaining = response.headers.get("Remaining-Req", "")
        match = re.search(r"\bsec=(\d+)", remaining)
        if match and int(match[1]) <= 1:
            time.sleep(1)

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected API response: {payload!r}")
        return payload

    raise RuntimeError("Request attempts exhausted.")


def collect(
    client: httpx.Client,
    market: str,
    minutes: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Collect candles whose opening timestamps fall within [start, end).

    Both bounds must be UTC-aware. The caller is responsible for aligning
    them to candle boundaries and excluding unfinished candles.

    Pagination moves backward using the oldest timestamp in each page.
    Results are deduplicated by timestamp and returned in ascending order.

    Warning:
    An empty API page ends collection, even if start has not been reached.
    Consequently, a successful return does not guarantee complete coverage.

    Raises:
        RuntimeError: If pagination fails to move backward.
        httpx.HTTPError: If a page request fails.
    """
    cursor = end
    candles: dict[datetime, dict] = {}
    pages = 0

    while cursor > start:
        page = request_page(client, market, minutes, cursor)
        if not page:
            break

        times = [candle_time(candle) for candle in page]
        oldest = min(times)

        # Protect against an unexpected API response causing an infinite loop.
        if oldest >= cursor:
            raise RuntimeError("Pagination made no progress; stopping.")

        for timestamp, candle in zip(times, page):
            if start <= timestamp < end:
                candles[timestamp] = candle

        pages += 1
        print(
            f"Pages: {pages:>4} | Candles: {len(candles):>7,} "
            f"| Reached: {iso_utc(oldest)}",
            flush=True,
        )

        # Upbit's "to" boundary is exclusive, so no time subtraction is needed.
        cursor = oldest

    return [candles[timestamp] for timestamp in sorted(candles)]


def save_csv(
    candles: list[dict],
    output: Path,
    market: str,
    minutes: int,
) -> None:
    """Write candles to a UTF-8 CSV without changing their input order.

    Prices and volumes are copied from the API without intentional rounding.
    The supplied market and interval are recorded on every row.

    Write to a sibling .part file first, then replace the destination after
    the file closes successfully. A failed write may leave the .part file;
    it is not a resumable download checkpoint.

    Raises:
        OSError: If directory creation, writing, or replacement fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")

    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for candle in candles:
            writer.writerow({
                "market": market,
                "interval_minutes": minutes,
                "time_utc": iso_utc(candle_time(candle)),
                "open": candle["opening_price"],
                "high": candle["high_price"],
                "low": candle["low_price"],
                "close": candle["trade_price"],
                "volume": candle["candle_acc_trade_volume"],
                "quote_volume": candle["candle_acc_trade_price"],
            })

    temporary.replace(output)


def main() -> None:
    """Parse CLI options, collect completed candles, and report coverage.

    Relative output paths are resolved from the current working directory.
    Defaults collect 90 days of KRW-BTC one-minute candles into data/raw.
    """
    parser = argparse.ArgumentParser(
        description="Download completed Upbit minute candles as CSV."
    )
    parser.add_argument("--market", default="KRW-BTC")
    parser.add_argument(
        "--interval", type=parse_interval, default=parse_interval("1m")
    )
    parser.add_argument(
        "--period", type=parse_duration, default=parse_duration("90d")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw")
    )
    args = parser.parse_args()

    # Validate syntax locally; market availability is checked by the API.
    market = args.market.upper()
    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", market):
        parser.error("Market must look like KRW-BTC or KRW-ETH.")

    step_seconds = args.interval * 60
    period_seconds = int(args.period.total_seconds())
    if period_seconds % step_seconds:
        parser.error("--period must be a multiple of --interval.")

    # Floor the execution time to the current candle's opening boundary.
    # Excluding that boundary removes the unfinished candle from the window.
    now = datetime.now(UTC)
    end = datetime.fromtimestamp(
        int(now.timestamp()) // step_seconds * step_seconds,
        tz=UTC,
    )
    start = end - args.period

    print(f"Market:   {market}")
    print(f"Interval: {args.interval} minute(s)")
    print(f"Range:    [{iso_utc(start)}, {iso_utc(end)})")

    try:
        with httpx.Client(timeout=30) as client:
            candles = collect(
                client, market, args.interval, start, end
            )

        if not candles:
            raise RuntimeError("No candles returned for the requested range.")

        # Include both the requested window and execution timestamp so that
        # repeated downloads normally produce distinct filenames.
        filename = (
            f"{market}_{args.interval}m_"
            f"{start:%Y%m%dT%H%M%SZ}_"
            f"{end:%Y%m%dT%H%M%SZ}_"
            f"{now:%Y%m%dT%H%M%S%fZ}.csv"
        )
        output = args.output_dir / filename
        save_csv(candles, output, market, args.interval)

    except (httpx.HTTPError, RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Collection failed: {error}\n")

    # Compare returned rows with the theoretical number of time slots.
    # This reports absent candles without determining their cause:
    # no trades, limited historical coverage, or another data issue.
    expected = period_seconds // step_seconds
    missing = expected - len(candles)

    print(f"\nSaved: {output.resolve()}")
    print(f"Rows: {len(candles):,} / Time slots: {expected:,}")
    print(f"First: {iso_utc(candle_time(candles[0]))}")
    print(f"Last:  {iso_utc(candle_time(candles[-1]))}")
    if missing:
        print(
            f"WARNING: {missing:,} time slots have no returned candle. "
            "No synthetic candles were inserted."
        )


if __name__ == "__main__":
    main()