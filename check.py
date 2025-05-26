import csv
import argparse
from tqdm import tqdm

def classify_membership(input_csv, output_csv, ratio_threshold, zlib_threshold):
    """
    Reads the results CSV and flags each sample as "member" or "non-member"
    based on:
      - ppl_ratio > ratio_threshold
      - zlib_ratio < zlib_threshold
    Writes a new CSV with an additional 'is_member' column.
    """
    with open(input_csv, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['is_member']
        rows = []
        for row in reader:
            try:
                ratio = float(row['ppl_ratio'])
                zratio = float(row['zlib_ratio'])
            except ValueError:
                # Skip rows with malformed numbers
                continue
            is_member = (ratio > ratio_threshold and zratio < zlib_threshold)
            row['is_member'] = str(is_member)
            rows.append(row)

    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify samples as members based on thresholds'
    )
    parser.add_argument('--input', type=str, default='results.csv',
                        help='Path to input results CSV')
    parser.add_argument('--output', type=str, default='flagged_results.csv',
                        help='Path to output CSV with membership flags')
    parser.add_argument('--ratio-threshold', type=float, default=1.1,
                        help='Minimum ppl_ratio to flag membership')
    parser.add_argument('--zlib-threshold', type=float, default=0.8,
                        help='Maximum zlib_ratio to flag membership')
    args = parser.parse_args()

    print(f"Classifying with ratio > {args.ratio_threshold} and zlib < {args.zlib_threshold}")
    classify_membership(args.input, args.output,
                        args.ratio_threshold, args.zlib_threshold)
    print(f"Done. Flagged results written to {args.output}")
