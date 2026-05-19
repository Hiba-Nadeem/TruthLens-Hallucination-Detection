import argparse
import csv
from pathlib import Path


KEEP_LABELS = {"SUPPORTS", "REFUTES"}


def filter_rows(input_path: Path, output_path: Path, label_column: str) -> None:
    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        if label_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{label_column}' not found. Available columns: {reader.fieldnames}"
            )

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                if row[label_column].strip().upper() in KEEP_LABELS:
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter FEVER CSV rows to keep only Supported and Refutes labels."
    )
    parser.add_argument(
        "--input",
        default="../data/fever_500.csv",
        help="Path to the input CSV file. Default: fever_500.csv",
    )
    parser.add_argument(
        "--output",
        default="../data/fever_500_filtered.csv",
        help="Path to the output CSV file. Default: fever_500_filtered.csv",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column. Default: label",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    filter_rows(input_path, output_path, args.label_column)
    print(f"Filtered file saved to: {output_path}")


if __name__ == "__main__":
    main()
