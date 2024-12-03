import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '.\\Results.xlsx'  # Replace with your file path
sheet_name = "Speed_up"  # Specify the sheet name

# Read the data
data = pd.read_excel(file_path, sheet_name=sheet_name)

# Ensure the columns "Model Name" and "Speed up" exist
if "Model" in data.columns and "Speedup" in data.columns:
    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(data["Model"], data["Speedup"], color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Speedup")
    plt.title("Model Speed Up Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig("Adaptivity_BarGraph.png")  # Saves the graph as a PNG file
else:
    print("The required columns 'Model Name' and 'Speed up' are not found in the sheet.")

plt.savefig("Speed_up.png")  # Saves the graph as a PNG file