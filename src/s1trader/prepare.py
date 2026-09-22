"""Fill short candle gaps while preserving long gaps and original values.

Synthetic candles use the previous close for OHLC and zero for both
volume fields. They are explicitly marked as non-tradable.

Long gaps remain absent from the output. The next observed candle starts
a new segment, but this module does not reset indicators or positions.

The input is validated before processing. Existing validation warnings
are allowed; errors stop processing. The source file is never modified.

Examples:
    uv run python -m s1trader.prepare data/raw/example.csv
    uv run python -m s1trader.prepare data/raw/example.csv --max-fill 2
    uv run python -m s1trader.prepare data/raw/example.csv --max-fill 0
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from .validate import Report, parse_timestamp, validate


UTC = timezone.utc
ADDED_FIELDS = [
    "is_synthetic",
    "can_trade",
    "segment_id",
    "gap_before_minutes",
]


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer for an argparse option."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected a nonnegative integer."
        ) from error

    if number < 0:
        raise argparse.ArgumentTypeError(
            "Expected a nonnegative integer."
        )
    return number


def iso_utc(value: datetime) -> str:
    """Format an aware timestamp as an ISO 8601 UTC string."""
    return value.astimezone(UTC).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def annotate(
    row: dict[str, str],
    *,
    synthetic: bool,
    segment: int,
    gap_minutes: int = 0,
) -> dict[str, str]:
    """Copy a candle and append preparation metadata.

    can_trade is a bar-level simulation eligibility flag, not a guarantee
    of sufficient liquidity or execution at a particular price.
    """
    can_trade = (
        not synthetic
        and Decimal(row["volume"]) > 0
        and Decimal(row["quote_volume"]) > 0
    )

    return {
        **row,
        "is_synthetic": str(synthetic).lower(),
        "can_trade": str(can_trade).lower(),
        "segment_id": str(segment),
        "gap_before_minutes": str(gap_minutes),
    }


def prepare(
    source: Path,
    output_dir: Path,
    max_fill: int,
) -> Path:
    """Validate raw candles, fill short gaps, and save a separate CSV.

    max_fill is the maximum number of consecutive missing candles to
    synthesize. Zero disables filling. A longer gap is preserved in full;
    it is never partially filled.

    Only gaps between observed records are considered. Leading or trailing
    candles are not invented.

    The source is expected to remain unchanged throughout processing.

    Raises:
        ValueError: If validation fails or preparation columns already exist.
        OSError: If reading or writing fails.
        csv.Error: If CSV parsing fails.
    """
    report = Report()
    validate(source, report)
    report.display()

    if report.errors:
        raise ValueError(
            f"Input validation found {report.errors} error(s). "
            "Fix them before preparing the data."
        )

    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    # Prevent accidentally preparing an already prepared file.
    collisions = set(fields) & set(ADDED_FIELDS)
    if collisions:
        raise ValueError(
            "Input already contains preparation columns: "
            + ", ".join(sorted(collisions))
        )

    minutes = int(rows[0]["interval_minutes"])
    step = timedelta(minutes=minutes)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents an existing result from being overwritten.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"{source.stem}_prepared_{stamp}.csv"
    temporary = output.with_suffix(".csv.part")

    segment = 0
    synthetic_count = 0
    preserved_gaps = 0
    preserved_minutes = 0
    original_nontradable = 0
    previous: dict[str, str] | None = None
    previous_time: datetime | None = None

    with temporary.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields + ADDED_FIELDS,
        )
        writer.writeheader()

        for row in rows:
            timestamp = parse_timestamp(row["time_utc"])
            gap_minutes = 0

            if previous is not None and previous_time is not None:
                missing = (timestamp - previous_time) // step - 1

                if 0 < missing <= max_fill:
                    for offset in range(1, missing + 1):
                        # Unknown extra columns remain empty rather than
                        # inheriting potentially invalid prior-row values.
                        synthetic = dict.fromkeys(fields, "")
                        synthetic.update({
                            "market": row["market"],
                            "interval_minutes": row["interval_minutes"],
                            "time_utc": iso_utc(
                                previous_time + offset * step
                            ),
                            "open": previous["close"],
                            "high": previous["close"],
                            "low": previous["close"],
                            "close": previous["close"],
                            "volume": "0",
                            "quote_volume": "0",
                        })
                        writer.writerow(annotate(
                            synthetic,
                            synthetic=True,
                            segment=segment,
                        ))
                        synthetic_count += 1

                elif missing > max_fill:
                    # Record only the absent duration, excluding the
                    # interval occupied by the previous observed candle.
                    gap_minutes = missing * minutes
                    segment += 1
                    preserved_gaps += 1
                    preserved_minutes += gap_minutes

            prepared = annotate(
                row,
                synthetic=False,
                segment=segment,
                gap_minutes=gap_minutes,
            )
            writer.writerow(prepared)

            if prepared["can_trade"] == "false":
                original_nontradable += 1

            previous = row
            previous_time = timestamp

    # Publish the final filename only after the CSV is fully written.
    temporary.replace(output)

    print("\nPreparation complete")
    print(f"Original candles:       {len(rows):,}")
    print(f"Synthetic candles:      {synthetic_count:,}")
    print(f"Output candles:         {len(rows) + synthetic_count:,}")
    print(f"Preserved gap runs:     {preserved_gaps:,}")
    print(f"Preserved gap minutes:  {preserved_minutes:,}")
    print(f"Segments:               {segment + 1:,}")
    print(f"Non-tradable originals: {original_nontradable:,}")
    print(f"Saved: {output.resolve()}")

    return output


def main() -> None:
    """Parse CLI options and prepare one raw candle CSV."""
    parser = argparse.ArgumentParser(
        description="Fill short gaps and annotate candle trading eligibility."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--max-fill",
        type=nonnegative_int,
        default=1,
        help=(
            "Maximum consecutive missing candles to fill, not minutes. "
            "Default: 1. Use 0 to preserve all gaps."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared"),
    )
    args = parser.parse_args()

    try:
        prepare(args.csv_path, args.output_dir, args.max_fill)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()