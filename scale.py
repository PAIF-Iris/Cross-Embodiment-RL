import pandas as pd
import numpy as np

# Load the parquet file
parquet_path = "/Users/paif_iris/Desktop/metaworld/episode_000000.parquet"  # UPDATE THIS PATH
df = pd.read_parquet(parquet_path)

print("="*70)
print("DETAILED ACTION ANALYSIS")
print("="*70)
print(f"Total timesteps: {len(df)}\n")

# Extract all actions
all_actions = np.array([df['action'].iloc[i] for i in range(len(df))])

print("7D Action breakdown:")
print(f"  Dimensions: {all_actions.shape}")
print(f"  [0:3] = dx, dy, dz (position deltas)")
print(f"  [3:6] = droll, dpitch, dyaw (orientation deltas)")
print(f"  [6]   = gripper\n")

# Analyze each dimension
dimension_names = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'gripper']

print("Statistics per dimension:")
print(f"{'Dim':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'Mean|abs|':<10}")
print("-"*70)

for i, name in enumerate(dimension_names):
    dim_data = all_actions[:, i]
    print(f"{name:<10} {dim_data.min():<10.4f} {dim_data.max():<10.4f} "
          f"{dim_data.mean():<10.4f} {dim_data.std():<10.4f} "
          f"{np.mean(np.abs(dim_data)):<10.4f}")

print("\n" + "="*70)
print("POSITION DELTA ANALYSIS")
print("="*70)

pos_deltas = all_actions[:, :3]
pos_magnitudes = np.linalg.norm(pos_deltas, axis=1)

print(f"Position delta magnitudes (||[dx, dy, dz]||):")
print(f"  Min:  {pos_magnitudes.min():.6f}")
print(f"  Max:  {pos_magnitudes.max():.6f}")
print(f"  Mean: {pos_magnitudes.mean():.6f}")
print(f"  Std:  {pos_magnitudes.std():.6f}")

# Calculate total distance traveled
total_distance = np.sum(pos_magnitudes)
print(f"\nTotal end-effector distance traveled: {total_distance:.4f} meters")
print(f"Average speed: {total_distance / df['timestamp'].iloc[-1]:.4f} m/s")

print("\n" + "="*70)
print("GRIPPER ANALYSIS")
print("="*70)

gripper = all_actions[:, 6]
unique_gripper = np.unique(gripper)

print(f"Gripper values:")
print(f"  Number of unique values: {len(unique_gripper)}")
print(f"  Values: {unique_gripper}")
print(f"  Distribution:")

for val in unique_gripper:
    count = np.sum(gripper == val)
    print(f"    {val:.3f}: {count} times ({100*count/len(gripper):.1f}%)")

# Detect gripper state changes
gripper_changes = np.where(np.diff(gripper) != 0)[0]
print(f"\n  Gripper state changes: {len(gripper_changes)} times")
if len(gripper_changes) > 0:
    print(f"  Change timesteps: {gripper_changes[:10]}...")  # Show first 10

print("\n" + "="*70)
print("SCALING RECOMMENDATIONS")
print("="*70)

# MetaWorld typically expects actions in [-1, 1] range
# Calculate what scale would bring actions into reasonable range
max_pos_delta = np.max(np.abs(pos_deltas))
recommended_scale = 1.0 / max_pos_delta

print(f"\nMetaWorld expects actions in range [-1, 1]")
print(f"Your maximum position delta: {max_pos_delta:.6f}")
print(f"\nRecommended scaling factors to test:")
print(f"  Conservative (50% of max): {recommended_scale * 0.5:.1f}")
print(f"  Moderate (80% of max):     {recommended_scale * 0.8:.1f}")
print(f"  Aggressive (100% of max):  {recommended_scale:.1f}")
print(f"\nSuggestion: Start with scale = {recommended_scale * 0.5:.1f} and adjust based on robot movement")

print("\n" + "="*70)
print("TRAJECTORY VISUALIZATION")
print("="*70)

print("\nFirst 10 actions:")
for i in range(min(10, len(df))):
    action = all_actions[i]
    print(f"  Step {i}: pos=[{action[0]:7.4f}, {action[1]:7.4f}, {action[2]:7.4f}] "
          f"rot=[{action[3]:7.4f}, {action[4]:7.4f}, {action[5]:7.4f}] grip={action[6]:.3f}")

print("\n" + "="*70)
print(f"Analysis complete! Use recommended scale factor in replay script.")
print("="*70)