import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to the folder containing the .mat files
folder_path = '../data/30'

# Get the first .mat file in the folder
mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
if not mat_files:
    print("No .mat files found in the specified folder.")
    exit()

file_path = os.path.join(folder_path, mat_files[0])
print(f"Inspecting file: {file_path}")

# Load the .mat file
mat_data = scipy.io.loadmat(file_path)

# Print the keys in the .mat file
print("\nKeys in the .mat file:")
print(mat_data.keys())

# Assuming 'inputfeats_ph' is the key for your features
if 'inputfeats_ph' not in mat_data:
    print("'inputfeats_ph' not found in the .mat file. Please check the correct key.")
    exit()

features = mat_data['inputfeats_ph']
print(features)

# Print information about the features
print(f"\nShape of features: {features.shape}")
print(f"Data type of features: {features.dtype}")
print(f"Minimum value: {np.min(features)}")
print(f"Maximum value: {np.max(features)}")
print(f"Mean value: {np.mean(features)}")
print(f"Standard deviation: {np.std(features)}")

# Reshape the data to 2D for visualization
reshaped_features = features.reshape(-1, features.shape[-1])

# Plot a heatmap of the reshaped features
plt.figure(figsize=(10, 6))
plt.imshow(reshaped_features, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title("Heatmap of reshaped features")
plt.xlabel("Feature index")
plt.ylabel("Sample index")
plt.show()

# Plot histogram of feature values
plt.figure(figsize=(10, 6))
plt.hist(features.flatten(), bins=50)
plt.title("Histogram of feature values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Print first few elements
print("\nFirst few elements of the features:")
print(features[:2, :2, :, :2])

# Plot the first sample across all frequencies for the first feature
plt.figure(figsize=(10, 6))
plt.plot(features[0, :, 0, 0])
plt.title("First sample across all frequencies (first feature)")
plt.xlabel("Frequency bin")
plt.ylabel("Value")
plt.show()