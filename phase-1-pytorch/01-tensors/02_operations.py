import torch

# Create two tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations (each element with corresponding element)
print("Addition:", a + b)  # [5, 7, 9]
print("Subtraction:", a - b)  # [-3, -3, -3]
print("Multiplication:", a * b)  # [4, 10, 18]
print("Division:", a / b)  # [0.25, 0.4, 0.5]
print("Power:", a**2)  # [1, 4, 9]


# ============================================
# MATRIX MULTIPLICATION - The Heart of Deep Learning
# ============================================

# Let's start simple: 2D matrices
A = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape: (3, 2) → 3 rows, 2 columns

B = torch.tensor([[7, 8, 9], [10, 11, 12]])  # Shape: (2, 3) → 2 rows, 3 columns

# Matrix multiplication
C = torch.matmul(A, B)  # Can also write: A @ B

print(f"A shape: {A.shape}")  # (3, 2)
print(f"B shape: {B.shape}")  # (2, 3)
print(f"C shape: {C.shape}")  # (3, 3)
print(f"Result:\n{C}")


# ============================================
# BROADCASTING - Operations with Different Shapes
# ============================================

# Scenario: Add a bias to every row of a matrix

data = torch.tensor(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
)  # Shape: (3, 4)

# Bias: One value per feature
bias = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Shape: (4, )

# This works! PyTorch "broadcasts " bias to match data's shape

result = data + bias

print(f"Data shape: {data.shape}")  # (3, 4)
print(f"Bias shape: {bias.shape}")  # (4,)
print(f"Result shape: {result.shape}")  # (3, 4)
print(f"Result:\n{result}")

# Scaling Rows

# Multiple each row by a different scalar

data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape : (2, 3)

# Scale factors: one per row
scales = torch.tensor([[2.0], [3.0]])  # Shape: (2, 1)

result = data * scales
print(f"Result: \n{result}")

# [[2.0, 4.0, 6.0],    ← row 0 × 2
#  [12.0, 15.0, 18.0]] ← row 1 × 3

# Example 1: Add bonus to each student's scores 

print("=== Example 1: Bonus Points === ")

scores = torch.tensor([[80, 70, 90, 85],
                       [60, 75, 80, 70],
                       [90, 95, 85, 80]], dtype=torch.float32)

bonus = torch.tensor([5, 2, 3, 1], dtype=torch.float32)

print(f"Scores shapes: {scores.shape}") # (3, 4)
print(f"Bonus shape: {bonus.shape}") # (4, )

result = scores + bonus

print(f"Result shape: {result.shape}")  # (3, 4)
print(f"Result:\n{result}")

print("\n=== Example 2: Scale Each Row ===")
# Multiply each row by a different factor
data = torch.tensor([[1, 2, 3, 4],
                     [5, 6, 7, 8]], dtype=torch.float32) # (2, 4)

row_scale = torch.tensor([[10],
                          [20]], dtype=torch.float32) # (2, 1)

print(f"Data shape: {data.shape}") # (2, 4)
print("\n=== Example 2: Scale Each Row ===")

result2 = data * row_scale
print(f"Result shape: {result2.shape}")   # (2, 4)
print(f"Result:\n{result2}")
