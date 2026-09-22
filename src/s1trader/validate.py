"""Validate a candle CSV produced by s1trader.collect.

The source file is never modified. Validation covers schema, numeric
values, OHLC consistency, timestamps, duplicates, ordering, and internal
time gaps.

Coverage checks span only the earliest and latest valid timestamps.
Missing data before or after those bounds cannot be detected without an
external specification of the requested collection window.

Exit codes:
    0: No errors; warnings are allowed unless --strict is enabled.
    1: Validation failed.
    2: Invalid CLI arguments or an unreadable input file.

Examples:
    uv run python -m s1trader.validate data/raw/example.csv
    uv run python -m s1trader.validate data/raw/example.csv --strict
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


UTC = timezone.utc
SUPPORTED_MINUTES = {1, 3, 5, 10, 15, 30, 60, 240}
PRICE_FIELDS = ("open", "high", "low", "close")
VOLUME_FIELDS = ("volume", "quote_volume")
NUMERIC_FIELDS = PRICE_FIELDS + VOLUME_FIELDS
REQUIRED_FIELDS = {
    "market",
    "interval_minutes",
    "time_utc",
    *NUMERIC_FIELDS,
}


@dataclass
class Report:
    """Count every issue while retaining only a few examples per category."""

    max_examples: int = 3
    counts: Counter = field(default_factory=Counter)
    examples: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def add(self, severity: str, category: str, detail: str) -> None:
        """Record an issue without allowing diagnostics to grow unbounded."""
        key = (severity, category)
        self.counts[key] += 1
        samples = self.examples.setdefault(key, [])
        if len(samples) < self.max_examples:
            samples.append(detail)

    @property
    def errors(self) -> int:
        """Return the number of error findings, not the number of bad rows."""
        return sum(
            count
            for (severity, _), count in self.counts.items()
            if severity == "ERROR"
        )

    @property
    def warnings(self) -> int:
        """Return the number of warning findings."""
        return sum(
            count
            for (severity, _), count in self.counts.items()
            if severity == "WARNING"
        )

    def display(self) -> None:
        """Print category totals and a bounded selection of examples."""
        for (severity, category), count in sorted(self.counts.items()):
            print(f"\n{severity}: {category} ({count:,})")
            for example in self.examples[(severity, category)]:
                print(f"  {example}")


def parse_timestamp(value: str) -> datetime:
    """Parse a timezone-aware UTC timestamp with whole-second precision.

    Non-UTC offsets are rejected rather than silently converted because
    the collector explicitly promises UTC timestamps.
    """
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("timezone is missing")
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("timestamp must use UTC")
    if timestamp.microsecond:
        raise ValueError("fractional seconds are not allowed")
    return timestamp.astimezone(UTC)


def parse_number(
    value: str,
    name: str,
    row_number: int,
    report: Report,
) -> Decimal | None:
    """Read a finite numeric value without binary floating-point rounding."""
    try:
        number = Decimal(value)
    except InvalidOperation:
        report.add(
            "ERROR", "Invalid numeric value",
            f"Row {row_number}: {name}={value!r}",
        )
        return None

    if not number.is_finite():
        report.add(
            "ERROR", "Non-finite numeric value",
            f"Row {row_number}: {name}={value!r}",
        )
        return None

    return number


def validate(path: Path, report: Report) -> None:
    """Inspect CSV records and print data coverage statistics.

    CSV row numbers refer to logical records, with the header as row 1.
    Gap detection uses sorted unique timestamps, independently of input
    order. It is skipped when the candle interval is ambiguous or any
    timestamp cannot be placed reliably on the expected time grid.
    """
    timestamps: dict[datetime, int] = {}
    markets: set[str] = set()
    intervals: set[int] = set()
    previous: datetime | None = None
    timeline_valid = True
    rows = 0

    # utf-8-sig also accepts files containing a UTF-8 byte-order mark.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        headers = reader.fieldnames

        if not headers:
            report.add("ERROR", "Missing header", "The CSV has no header.")
            return

        if len(headers) != len(set(headers)):
            report.add(
                "ERROR", "Duplicate column names",
                "Header names must be unique.",
            )
            return

        missing = REQUIRED_FIELDS - set(headers)
        if missing:
            report.add(
                "ERROR", "Missing columns", ", ".join(sorted(missing))
            )
            return

        for row_number, row in enumerate(reader, start=2):
            rows += 1

            if None in row:
                report.add(
                    "ERROR", "Extra CSV fields",
                    f"Row {row_number}: more values than header columns.",
                )

            values = {
                name: (row.get(name) or "").strip()
                for name in REQUIRED_FIELDS
            }
            empty = [name for name, value in values.items() if not value]
            if empty:
                report.add(
                    "ERROR", "Empty required values",
                    f"Row {row_number}: {', '.join(sorted(empty))}",
                )

            if values["market"]:
                markets.add(values["market"])

            minutes = None
            try:
                minutes = int(values["interval_minutes"])
                if minutes not in SUPPORTED_MINUTES:
                    raise ValueError("unsupported interval")
                intervals.add(minutes)
            except ValueError:
                timeline_valid = False
                report.add(
                    "ERROR", "Invalid interval",
                    f"Row {row_number}: "
                    f"{values['interval_minutes']!r}",
                )
                minutes = None

            try:
                timestamp = parse_timestamp(values["time_utc"])
            except ValueError as error:
                timeline_valid = False
                report.add(
                    "ERROR", "Invalid timestamp",
                    f"Row {row_number}: {error}",
                )
            else:
                if minutes is not None:
                    if int(timestamp.timestamp()) % (minutes * 60):
                        timeline_valid = False
                        report.add(
                            "ERROR", "Misaligned timestamp",
                            f"Row {row_number}: {timestamp.isoformat()} "
                            f"is not on a {minutes}-minute boundary.",
                        )

                if timestamp in timestamps:
                    report.add(
                        "ERROR", "Duplicate timestamp",
                        f"Rows {timestamps[timestamp]} and {row_number}: "
                        f"{timestamp.isoformat()}",
                    )
                else:
                    timestamps[timestamp] = row_number

                if previous is not None and timestamp < previous:
                    report.add(
                        "ERROR", "Out-of-order timestamp",
                        f"Row {row_number}: timestamp precedes the "
                        "previous record.",
                    )
                previous = timestamp

            numbers = {
                name: parse_number(values[name], name, row_number, report)
                for name in NUMERIC_FIELDS
                if values[name]
            }

            for name in PRICE_FIELDS:
                value = numbers.get(name)
                if value is not None and value <= 0:
                    report.add(
                        "ERROR", "Non-positive price",
                        f"Row {row_number}: {name}={value}",
                    )

            for name in VOLUME_FIELDS:
                value = numbers.get(name)
                if value is not None and value < 0:
                    report.add(
                        "ERROR", "Negative volume",
                        f"Row {row_number}: {name}={value}",
                    )
                elif value == 0:
                    report.add(
                        "WARNING", "Zero volume",
                        f"Row {row_number}: {name}=0",
                    )

            if all(numbers.get(name) is not None for name in PRICE_FIELDS):
                opening = numbers["open"]
                high = numbers["high"]
                low = numbers["low"]
                close = numbers["close"]

                if not (low <= opening <= high and low <= close <= high):
                    report.add(
                        "ERROR", "Inconsistent OHLC",
                        f"Row {row_number}: open={opening}, high={high}, "
                        f"low={low}, close={close}",
                    )

    print(f"Rows:              {rows:,}")
    print(f"Unique timestamps: {len(timestamps):,}")
    print(f"Markets:           {', '.join(sorted(markets)) or '(none)'}")
    print(f"Intervals:         {sorted(intervals)} minute(s)")

    if not rows:
        report.add("ERROR", "Empty dataset", "No data records were found.")

    if len(markets) > 1:
        report.add(
            "ERROR", "Mixed markets",
            "A candle file must contain exactly one market.",
        )

    if len(intervals) > 1:
        report.add(
            "ERROR", "Mixed intervals",
            "A candle file must contain exactly one interval.",
        )

    ordered = sorted(timestamps)
    if ordered:
        print(f"First timestamp:   {ordered[0].isoformat()}")
        print(f"Last timestamp:    {ordered[-1].isoformat()}")

    if not timeline_valid or len(intervals) != 1 or not ordered:
        print("Gap detection:     skipped due to invalid timeline metadata")
        return

    step = timedelta(minutes=next(iter(intervals)))
    missing_slots = 0

    # Inspect adjacent timestamps instead of allocating a full time grid.
    for left, right in zip(ordered, ordered[1:]):
        missing = (right - left) // step - 1
        if missing > 0:
            missing_slots += missing
            report.add(
                "WARNING", "Missing candle interval",
                f"{(left + step).isoformat()} through "
                f"{(right - step).isoformat()}: "
                f"{missing:,} missing candle(s)",
            )

    print(f"Missing slots:     {missing_slots:,}")


def main() -> None:
    """Run validation and expose its outcome through the process exit code."""
    parser = argparse.ArgumentParser(
        description="Validate a candle CSV without modifying it."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as a failed validation.",
    )
    args = parser.parse_args()

    report = Report()
    print(f"File: {args.csv_path.resolve()}\n")

    try:
        validate(args.csv_path, report)
    except (OSError, UnicodeError, csv.Error) as error:
        parser.exit(2, f"Cannot read CSV: {error}\n")

    report.display()

    failed = report.errors > 0 or (args.strict and report.warnings > 0)
    status = "FAIL" if failed else (
        "PASS WITH WARNINGS" if report.warnings else "PASS"
    )

    print(
        f"\nResult: {status} "
        f"| Errors: {report.errors:,} "
        f"| Warnings: {report.warnings:,}"
    )
    parser.exit(1 if failed else 0)


if __name__ == "__main__":
    main()