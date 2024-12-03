import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '.\\Results.xlsx'  # Replace with the actual file path
excel_data = pd.ExcelFile(file_path)

# Read the Adaptivity sheet
adaptivity_data = excel_data.parse("Adaptivity")

# Extract data for the plot
models = adaptivity_data["Model Name"]
on_values = adaptivity_data["On"]
off_values = adaptivity_data["Off"]

# Create a stacked bar graph
plt.figure(figsize=(10, 6))

# Plot stacked bars
x_positions = range(len(models))
bar_width = 0.5
plt.bar(x_positions, on_values, bar_width, label='On', color='skyblue')
plt.bar(x_positions, off_values, bar_width, bottom=on_values, label='Off', color='orange')

# Add labels, title, and legend
plt.xlabel("Model Name")
plt.ylabel("Count")
plt.title("Adaptivity: On (bottom) vs Off (top)")
plt.xticks([x for x in x_positions], models, rotation=45)
plt.legend()

# Show and save the graph
plt.tight_layout()
plt.savefig("Adaptivity.png")  # Saves the graph as a PNG file
plt.show()
