import pandas as pd
import os

output_dir = "output"
benign_path = os.path.join(output_dir, "benign_labeled.csv")
malicious_path = os.path.join(output_dir, "malicious_labeled.csv")

benign_df = pd.read_csv(benign_path)
malicious_df = pd.read_csv(malicious_path)

combined_df = pd.concat([benign_df, malicious_df], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)
combined_df.to_csv(os.path.join(output_dir, "labeled_flows.csv"), index=False)
print("Merged labeled flows saved.")