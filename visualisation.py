import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize_membership(input_csv):
    df = pd.read_csv(input_csv)
    # Ensure is_member is boolean
    df['is_member'] = df['is_member'].astype(str).map({'True': True, 'False': False})

    total = len(df)
    member_count = df['is_member'].sum()
    non_member_count = total - member_count
    member_pct = member_count / total * 100

    print(f"Total samples: {total}")
    print(f"Members: {member_count} ({member_pct:.2f}%)")
    print(f"Non-members: {non_member_count} ({100-member_pct:.2f}%)")

    # Plot bar chart
    plt.figure()
    plt.bar(['Member', 'Non-member'], [member_count, non_member_count])
    plt.xlabel('Membership Status')
    plt.ylabel('Number of Samples')
    plt.title('Pre-training Data Membership')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute membership percentage and visualize results"
    )
    parser.add_argument(
        '--input', type=str, default='flagged_results.csv',
        help='Path to the flagged results CSV'
    )
    args = parser.parse_args()
    visualize_membership(args.input)
