import pandas as pd
import numpy as np
import os

# Define the number of samples
sample_size = 10000

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
data = {
    'Age': np.random.randint(18, 70, sample_size),
    'Annual Income': np.random.randint(20000, 150000, sample_size),
    'Credit Score': np.random.randint(300, 850, sample_size),
    'Loan Amount': np.random.uniform(5000, 50000, sample_size).round(2),
    'Existing Loan': np.random.randint(0, 2, sample_size),
    'Debt to Income': np.random.uniform(0.1, 1.0, sample_size).round(2),
    'Has Default': np.random.choice([0, 1], size=sample_size, p=[0.5, 0.5])  # Balanced classes
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the folder structure
folder_path = "data/raw"
os.makedirs(folder_path, exist_ok=True)

# Save the dataset to the folder
file_path = os.path.join(folder_path, "raw_credit_data.csv")
df.to_csv(file_path, index=False)

print(f"Dataset generated and saved to {file_path}")
print("\nClass distribution in the generated dataset:")
print(df['Has Default'].value_counts())

