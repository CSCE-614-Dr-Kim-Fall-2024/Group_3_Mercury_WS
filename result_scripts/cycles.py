import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the Excel file
file_path = '.\\Results.xlsx'  # Replace with your file path
sheet_name = "Computation_cycles"  # Specify the sheet name

# Read the data
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Data preparation
models = data['Model_Name']
baseline_cycles = data['Baseline_Cycles(x 10^6)']
mercury_conv_cycles = data['MERCURY_convolution_Cycles']
rpq_cycles = data['RPQ_cycles']

# X-axis positions for bars
x = np.arange(len(models))

# Bar width
bar_width = 0.4

# Create the plot
plt.figure(figsize=(12, 7))

# Plot baseline cycles (blue bars)
plt.bar(x - bar_width / 2, baseline_cycles, width=bar_width, label='Baseline Cycles', color='blue')

# Plot Mercury cycles with stacked sections (orange and yellow)
plt.bar(x + bar_width / 2, mercury_conv_cycles, width=bar_width, label='MERCURY Convolution Cycles', color='orange')
plt.bar(x + bar_width / 2, rpq_cycles, width=bar_width, bottom=mercury_conv_cycles, label='RPQ Cycles', color='yellow')

# Formatting the plot
plt.xticks(x, models, rotation=45, ha='right')
plt.xlabel("Model_Name")
plt.ylabel("Cycles (x 10^6)")
plt.title("Comparison of Baseline and Mercury Cycles with Breakdown")
plt.legend()
plt.tight_layout()

# Show the graph
plt.show()
plt.savefig("Computations_cyles.png")  # Saves the graph as a PNG file