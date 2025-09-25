import pandas as pd
import os

# Define directories
possible_input_dirs = ["data/flow_csv", "data"]
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Possible filenames (fallback to ones present in this repo)
benign_candidates = [
	"benign_flow_stats.csv",
	"2017-05-02_kali-normal22_flow_stats.csv",
]
malicious_candidates = [
	"webgoat_flow_stats.csv",
]

def find_file(candidates, search_dirs):
	for d in search_dirs:
		for name in candidates:
			path = os.path.join(d, name)
			if os.path.exists(path):
				return path
	return None

benign_path = find_file(benign_candidates, possible_input_dirs)
malicious_path = find_file(malicious_candidates, possible_input_dirs)

if benign_path is None:
	raise FileNotFoundError(f"No benign flow CSV found. Tried: {benign_candidates} in {possible_input_dirs}")
if malicious_path is None:
	raise FileNotFoundError(f"No malicious flow CSV found. Tried: {malicious_candidates} in {possible_input_dirs}")

# Read and label benign flows
benign_df = pd.read_csv(benign_path)
benign_df["Type"] = "Benign"

# Read and label malicious flows
malicious_df = pd.read_csv(malicious_path)
malicious_df["Type"] = "Malicious"

# Save labeled versions
benign_df.to_csv(os.path.join(output_dir, "benign_labeled.csv"), index=False)
malicious_df.to_csv(os.path.join(output_dir, "malicious_labeled.csv"), index=False)
print(f"Labeled CSVs saved to {output_dir} (benign: {os.path.basename(benign_path)}, malicious: {os.path.basename(malicious_path)})")