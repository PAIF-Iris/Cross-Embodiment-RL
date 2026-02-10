# import pandas as pd
# import numpy as np

# PARQUET_PATH = "/Users/paif_iris/Desktop/metaworld/episode_000042.parquet"

# # ---------------------------------------------------
# # 1. Load parquet
# # ---------------------------------------------------
# df = pd.read_parquet(PARQUET_PATH)

# print("\n=== BASIC INFO ===")
# print("Number of rows (timesteps):", len(df))
# print("Columns:\n", df.columns.tolist())

# print("\n=== DATAFRAME INFO ===")
# print(df.info())

# # ---------------------------------------------------
# # 2. Inspect each column: type + shape
# # ---------------------------------------------------
# print("\n=== COLUMN DETAILS ===")

# for col in df.columns:
#     sample = df[col].iloc[0]
#     print(f"\nColumn: {col}")
#     print("  Python type:", type(sample))

#     if isinstance(sample, (list, tuple, np.ndarray)):
#         arr = np.array(sample)
#         print("  Shape:", arr.shape)
#         print("  Dtype:", arr.dtype)
#         print("  First 5 values:", arr.flatten()[:5])
#     else:
#         print("  Value:", sample)

# # ---------------------------------------------------
# # 3. Identify likely state / action columns
# # ---------------------------------------------------
# print("\n=== LIKELY MODALITIES ===")

# for col in df.columns:
#     name = col.lower()
#     if "action" in name:
#         print("Action column:", col)
#     if "state" in name:
#         print("State column:", col)
#     if "image" in name or "camera" in name:
#         print("Image column:", col)

# # ---------------------------------------------------
# # 4. Explicit checks for Meta-World layout
# # ---------------------------------------------------
# if "observation.state" in df.columns:
#     state = np.array(df["observation.state"].iloc[0])
#     print("\n=== OBSERVATION.STATE ===")
#     print("State vector length:", state.shape)
#     print("EE pos (0:3):", state[0:3])
#     print("Gripper (3):", state[3])
#     print("Object pos (4:7):", state[4:7])
#     print("Object quat (7:11):", state[7:11])

# if "action" in df.columns:
#     action = np.array(df["action"].iloc[0])
#     print("\n=== ACTION ===")
#     print("Action shape:", action.shape)
#     print("Action:", action)

# # ---------------------------------------------------
# # 5. Head & tail preview
# # ---------------------------------------------------
# print("\n=== FIRST 3 ROWS ===")
# print(df.head(3))

# print("\n=== LAST 3 ROWS ===")
# print(df.tail(3))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


PARQUET_PATH = "/Users/paif_iris/Desktop/metaworld/episode_000042.parquet"

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
df = pd.read_parquet(PARQUET_PATH)
states = np.stack(df["observation.state"].to_numpy())  # (T, 81)

T, D = states.shape
timesteps = np.arange(T)

print(f"Loaded {T} timesteps, {D} state dimensions")

# ---------------------------------------------------
# Plot all dimensions
# ---------------------------------------------------
plt.figure(figsize=(14, 6))

for d in range(D):
    plt.plot(timesteps, states[:, d], alpha=0.4)

plt.xlabel("Timestep")
plt.ylabel("State value")
plt.title("observation.state (all dimensions)")
plt.tight_layout()
plt.show()
