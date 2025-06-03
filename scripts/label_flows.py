import pandas as pd
import os

# Define file paths
input_dir = "data/flow_csv"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

benign_path = os.path.join(input_dir, "benign_flow_stats.csv")
malicious_path = os.path.join(input_dir, "webgoat_flow_stats.csv")

# Read and label benign flows
benign_df = pd.read_csv(benign_path)
benign_df["Type"] = "Benign"

# Read and label malicious flows
malicious_df = pd.read_csv(malicious_path)
malicious_df["Type"] = "Malicious"

# Save labeled versions
benign_df.to_csv(os.path.join(output_dir, "benign_labeled.csv"), index=False)
malicious_df.to_csv(os.path.join(output_dir, "malicious_labeled.csv"), index=False)
print("Labeled CSVs saved.")