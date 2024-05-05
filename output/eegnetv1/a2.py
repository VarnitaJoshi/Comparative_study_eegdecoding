import pandas as pd
import os

# Initialize empty DataFrame to store extracted information
df = pd.DataFrame(columns=['Model', 'Accuracy'])

# Iterate over file names
file_name = 'training_log_eegnetv1try_exp_cleaned.txt'
with open(file_name, 'r') as file:
    lines = file.readlines()

# Extract valid_accuracy information
accuracies = []
for line in lines:
    if line.startswith('     ') and not line.startswith('    -'):
        parts = line.split()
        a = round(float(parts[4]) * 100, 3)
        accuracies.append(a)


# Append extracted data to DataFrame
data = {'Model': 'EEGNetv1', 'Accuracy': accuracies}
df = df.append(pd.DataFrame(data), ignore_index=True)

# Save DataFrame to CSV file
df.to_csv('a2.csv', index=False)
