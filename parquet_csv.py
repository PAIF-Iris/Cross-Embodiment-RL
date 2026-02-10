import pandas as pd
import numpy as np

PARQUET_PATH = "/Users/paif_iris/Desktop/metaworld/episode_000042.parquet"
CSV_PATH = "/Users/paif_iris/Desktop/metaworld/your_dataset.csv"

# ---------------------------------------------------
# Load parquet
# ---------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)

print("Original columns:")
print(df.columns.tolist())

# ---------------------------------------------------
# Expand observation.state
# ---------------------------------------------------
if "observation.state" in df.columns:
    states = np.stack(df["observation.state"].to_numpy())
    state_dim = states.shape[1]

    state_cols = {
        f"state_{i}": states[:, i]
        for i in range(state_dim)
    }

    df_state = pd.DataFrame(state_cols)
    df = pd.concat([df.drop(columns=["observation.state"]), df_state], axis=1)

# ---------------------------------------------------
# Expand action
# ---------------------------------------------------
if "action" in df.columns:
    actions = np.stack(df["action"].to_numpy())
    action_dim = actions.shape[1]

    action_cols = {
        f"action_{i}": actions[:, i]
        for i in range(action_dim)
    }

    df_action = pd.DataFrame(action_cols)
    df = pd.concat([df.drop(columns=["action"]), df_action], axis=1)

# ---------------------------------------------------
# Drop image / video columns (optional but recommended)
# ---------------------------------------------------
image_cols = [c for c in df.columns if "image" in c or "camera" in c]
df = df.drop(columns=image_cols, errors="ignore")

print("\nFinal columns written to CSV:")
print(df.columns.tolist())
print("Total columns:", len(df.columns))

# ---------------------------------------------------
# Save CSV
# ---------------------------------------------------
df.to_csv(CSV_PATH, index=False)

print(f"\nCSV written to: {CSV_PATH}")
