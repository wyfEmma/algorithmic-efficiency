import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

# Paths to CSV files
pytorch_csv = '/root/experiments/sfadamw6/ogbg_pytorch/trial_1/measurements.csv'
jax_csv = '/root/experiments/sfadamw7/ogbg_jax/trial_1/measurements.csv'

# Read CSVs
try:
    pytorch_df = pd.read_csv(pytorch_csv)
    print("PyTorch CSV columns:", pytorch_df.columns)
    jax_df = pd.read_csv(jax_csv)
    print("JAX CSV columns:", jax_df.columns)
except FileNotFoundError as e:
    print(f"Error: Could not find CSV file: {e}")
    exit()
except pd.errors.EmptyDataError as e:
    print(f"Error: CSV file is empty: {e}")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Define the correct column names based on inspection
x_column = 'global_step'
y_column = 'validation/loss'

# Create output directory in /tmp which is writable
output_dir = '/tmp/plots'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'adam_ogbg_plot.png')

# --- FIX: Clean the data by dropping rows with NaN values ---
pytorch_df_cleaned = pytorch_df.dropna(subset=[y_column])
jax_df_cleaned = jax_df.dropna(subset=[y_column])

# Print data after cleaning
print("\nCleaned PyTorch data:")
print(pytorch_df_cleaned[[x_column, y_column]].head())
print("\nCleaned JAX data:")
print(jax_df_cleaned[[x_column, y_column]].head())

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(pytorch_df_cleaned[x_column], pytorch_df_cleaned[y_column], label='PyTorch', marker='o')
plt.plot(jax_df_cleaned[x_column], jax_df_cleaned[y_column], label='JAX', marker='x')

plt.xlabel('Step')
plt.ylabel('Metric')
plt.title('Comparison of PyTorch and JAX Metrics')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
try:
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.close()

# Open the plot in the default browser (if possible)
try:
    subprocess.run(["$BROWSER", output_path], shell=True, check=True)
except FileNotFoundError:
    print("No browser found to open the image.")
except subprocess.CalledProcessError:
    print("Error opening the image in the browser.")
except Exception as e:
    print(f"Error opening plot in browser: {e}")