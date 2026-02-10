import pandas as pd
import pyarrow.parquet as pq

# Update this path to your actual .parquet file location
path = "/Users/paif_iris/Desktop/metaworld/episode_000000.parquet"  # <-- UPDATE THIS

# First, let's look at the parquet file metadata
parquet_file = pq.ParquetFile(path)
print("=== Parquet File Metadata ===")
print(f"Number of row groups: {parquet_file.num_row_groups}")
print(f"Schema:")
print(parquet_file.schema)
print("\n" + "="*70 + "\n")

# Load the data
df = pd.read_parquet(path)

print("=== DataFrame Info ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nColumn types:")
print(df.dtypes)
print("\n" + "="*70 + "\n")

# Show first few rows
print("=== First few rows ===")
print(df.head())
print("\n" + "="*70 + "\n")

# Check the structure of specific columns
print("=== Detailed column inspection ===\n")

for col in df.columns:
    print(f"Column: {col}")
    print(f"  Type: {df[col].dtype}")
    
    # Check if it contains arrays/lists
    first_val = df[col].iloc[0]
    print(f"  First value type: {type(first_val)}")
    
    if hasattr(first_val, 'shape'):
        print(f"  First value shape: {first_val.shape}")
    elif isinstance(first_val, (list, tuple)):
        print(f"  First value length: {len(first_val)}")
    
    print(f"  First value: {first_val}")
    print()

print("="*70)

# Specifically check for the expected modalities
print("\n=== Looking for expected modalities ===")
if 'action' in df.columns:
    print(f"✓ Found 'action' column")
    print(f"  Shape of first action: {df['action'].iloc[0].shape if hasattr(df['action'].iloc[0], 'shape') else 'N/A'}")

if 'observation.state' in df.columns:
    print(f"✓ Found 'observation.state' column")
    print(f"  Shape of first state: {df['observation.state'].iloc[0].shape if hasattr(df['observation.state'].iloc[0], 'shape') else 'N/A'}")

if 'observation.images.world_camera' in df.columns:
    print(f"✓ Found 'observation.images.world_camera' column")
    
if 'observation.images.hand_camera' in df.columns:
    print(f"✓ Found 'observation.images.hand_camera' column")