import pandas as pd

# Paths to CSV files
pytorch_csv = '/root/experiments/sfadamw6/ogbg_pytorch/trial_1/measurements.csv'
jax_csv = '/root/experiments/sfadamw6/ogbg_jax/trial_1/measurements.csv'

# Read CSVs
try:
    pytorch_df = pd.read_csv(pytorch_csv)
    jax_df = pd.read_csv(jax_csv)
except FileNotFoundError as e:
    print(f"Error: Could not find CSV file: {e}")
    exit()
except pd.errors.EmptyDataError as e:
    print(f"Error: CSV file is empty: {e}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Print the number of data points
print(f"Number of data points in PyTorch CSV: {len(pytorch_df)}")
print(f"Number of data points in JAX CSV: {len(jax_df)}")