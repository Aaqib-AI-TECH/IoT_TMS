from imblearn.under_sampling import RandomUnderSampler
from Functions.Pipeline import pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Define paths and dataset name
in_dir = "<path_to_pcap_files>"  "./data/pcaps/"
out_dir = "<output_directory>"    "./results"
dataset_name = "<dataset_name>"   "UNSW"
processed_csv_file = "<path_to_preprocessed_csv>"  # e.g., "./data/processed.csv"

# Run pipeline (merges PCAP parsing with preprocessed CSV)
df = pipeline(in_dir, out_dir, dataset_name, processed_csv_file)

# Optional: Undersampling to balance class distribution
sampling_strategy = {
    "ClassA": 100000,
    "ClassB": 50000,
    "ClassC": 20000,
    # Add or modify based on your class labels
}
rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(
    df.loc[:, df.columns != "attack_cat"],
    df.loc[:, "attack_cat"]
)
X_res["attack_cat"] = y_res
df_reduced = X_res

# Convert hex payloads to byte-level features (1500 bytes)
X_payload = [
    np.array(bytearray.fromhex(p)) for p in df_reduced["payload"].to_numpy()
]
for x in X_payload:
    x.resize(1500, refcheck=False)
X_payload = np.row_stack(X_payload)

# Create payload byte column names
payload_columns = [f"payload_byte_{i+1}" for i in range(1500)]
payload_df = pd.DataFrame(X_payload, columns=payload_columns)

# Append selected metadata features
payload_df[[
    "sttl", "total_len", "t_delta", "protocol_m",
    "Transmission_time_sec", "Throughput", "network_latency", "delay"
]] = df_reduced[[
    "sttl", "total_len", "t_delta", "protocol_m",
    "Transmission_time_sec", "Throughput", "network_latency", "delay"
]]

# Encode protocol
payload_df["protocol_m"] = LabelEncoder().fit_transform(df_reduced["protocol_m"])

# Reorder: move the appended features to the front
cols = payload_df.columns.tolist()
cols = cols[-8:] + cols[:-8]
final = payload_df[cols]

# Add label column
final["label"] = df_reduced["label"].to_numpy().reshape((-1, 1))

# Save final processed dataset
final.to_csv(f"{out_dir}/{dataset_name}_converted_data.csv", index=False)
