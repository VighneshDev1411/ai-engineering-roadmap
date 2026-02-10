import torch

# Way 1: From a Python list
tensor_from_list = torch.tensor([1, 2, 3, 4])
print(f"From list: {tensor_from_list}")
print(f"Shape: {tensor_from_list.shape}")  # torch.Size([4]) — a 1D tensor with 4 elements
print(f"Data type: {tensor_from_list.dtype}")  # torch.int64 by default for integers

print("---")

# Way 2: From a nested list (creates 2D tensor / matrix)
matrix = torch.tensor([[1, 2, 3], 
                        [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"Shape: {matrix.shape}")  # torch.Size([2, 3]) — 2 rows, 3 columns

print("---")

# Way 3: Tensors filled with specific values
zeros = torch.zeros(3, 4)      # 3x4 matrix of zeros
ones = torch.ones(2, 3)        # 2x3 matrix of ones
random = torch.rand(2, 3)      # 2x3 matrix of random values between 0 and 1

print(f"Zeros (3x4):\n{zeros}")
print(f"Ones (2x3):\n{ones}")
print(f"Random (2x3):\n{random}")


# By default, floats become torch.float32, integers become torch.int64
# But in deep learning, we often need to be explicit

# Float tensor (most common in deep learning)
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Float tensor dtype: {float_tensor.dtype}")  # torch.float32

# Explicitly specify dtype
float64_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
print(f"Float64 tensor dtype: {float64_tensor.dtype}")  # torch.float64

# Integer tensor
int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f"Int32 tensor dtype: {int_tensor.dtype}")  # torch.int32

# Convert between types
converted = int_tensor.float()  # Convert to float32
print(f"Converted dtype: {converted.dtype}")  # torch.float32