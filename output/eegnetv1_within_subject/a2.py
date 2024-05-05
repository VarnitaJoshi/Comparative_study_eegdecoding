import pandas as pd
import os

# Initialize empty DataFrame to store extracted information
df = pd.DataFrame(columns=['Subject', 'Model', 'Accuracy'])

# Iterate over file names
for subject_number in range(1, 10):  # Assuming you have files from 1 to 8
    file_name = f'trainin_log_subject_{subject_number}_cleaned.txt'
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Extract valid_accuracy information
    accuracies = []
    for line in lines:
        if line.startswith('     ') and not line.startswith('    -'):
            parts = line.split()
            a = round(float(parts[4]) * 100, 3)
            accuracies.append(a)

    # Create a list containing the subject number
    subjects = [str(subject_number)] * len(accuracies)

    # Append extracted data to DataFrame
    data = {'Subject': subjects, 'Model': 'EEGNetv1', 'Accuracy': accuracies}
    df = df.append(pd.DataFrame(data), ignore_index=True)

# Save DataFrame to CSV file
df.to_csv('a2.csv', index=False)
