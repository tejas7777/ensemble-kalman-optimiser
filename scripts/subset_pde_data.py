import h5py
import torch

# Load the .mat file using h5py
file_path = './dataset/pde_data.mat'
data = h5py.File(file_path, 'r')

# Extract the data
inputs = data['Input'][()]
outputs = data['Output'][()]

# Specify the subset size
subset_size = 1000

# Create a small subset
input_subset = inputs[:subset_size]
output_subset = outputs[:subset_size]

# Convert the subset to PyTorch tensors
input_tensor = torch.tensor(input_subset, dtype=torch.float32)
output_tensor = torch.tensor(output_subset, dtype=torch.float32)

# Print the shapes to verify
print('Input tensor shape:', input_tensor.shape)
print('Output tensor shape:', output_tensor.shape)

# Save tensors to a new file
torch.save({'input': input_tensor, 'output': output_tensor}, './dataset/pde_data_subset.pt')

# Close the HDF5 file
data.close()

print("Subset saved successfully to './dataset/pde_data_subset.pt'")
