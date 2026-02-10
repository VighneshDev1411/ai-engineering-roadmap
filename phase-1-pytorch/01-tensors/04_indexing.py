import torch
# ============================================
# 1D TENSOR INDEXING
# ============================================

# A simple 1D tensor (like a Python list)
x = torch.tensor([10, 20, 30, 40, 50, 60, 70])
print(f"Orignal: {x}")

# Access single element (0-indexed, lie Python)
print(f"\nFirst element x[0]: {x[0]}")      # 10
print(f"Third element x[2]: {x[2]}")        # 30
print(f"Last element x[-1]: {x[-1]}")       # 70
print(f"Second last x[-2]: {x[-2]}")  # 60

# Slicing: x[start:end] - includes start, excludes end 
print(f"\nx[1:4]: {x[1:4]}") # [20, 30, 40] - index 1, 2, 3
print(f"x[:3]: {x[:3]}") # [10, 20, 30] - first 3 
print(f"x[4:]: {x[4:]}") # [50, 60, 70] - from index  4 to end 
print(f"x[::2]: {x[::2]}") # [10, 30, 50, 70] - every 2nd element 


# ============================================
# 2D TENSOR INDEXING (Matrices)
# ============================================

# Imagine this is 4 students, 3 exam scores each
scores = torch.tensor([[85, 90, 78],    # Student 0
                       [92, 88, 95],    # Student 1
                       [70, 75, 80],    # Student 2
                       [88, 82, 91]])   # Student 3

print(f"Full data:\n{scores}")
print(f"Shape: {scores.shape}")  # (4, 3)

# Access single element: [row, column]
print(scores[0, 1])
print(scores[2, 2])

# Get entire row (one student's all scores)
print(scores[1])
print(scores[1, :])

# Get entire column (one exam's all scores)
print(scores[:, 0])
print(scores[:, 2])

# Slice rows and columns together 
print(f"\n First 2 students, first 2 exams: \n{scores[:2, :2]}")
# [[85, 90],
#  [92, 88]]

print(f"\nLast 2 students, all exams:\n{scores[2:, :]}")
# [[70, 75, 80],
#  [88, 82, 91]]


# The key pattern tensor[row_selection, column_selection]


# ============================================
# REAL SCENARIO: Batch Selection from Images
# ============================================

# Simulating a batch of 8 grayscale images, each 28x28 pixels
# Shape: (batch_size, height, width)
images = torch.rand(8, 28, 28)

print(f"Full batch shape: {images.shape}")  # (8, 28, 28)

# Get first image
first_image = images[0]
print(f"\nFirst image shape: {first_image.shape}")  # (28, 28)

# Get first 4 images (mini-batch)
mini_batch = images[:4]
print(f"Mini batch shape: {mini_batch.shape}")  # (4, 28, 28)

# Get specific pixel from all images (row 14, column 14 — center pixel)
center_pixels = images[:, 14, 14]
print(f"Center pixel of all images: {center_pixels.shape}")  # (8,)

# Get top-left 10x10 corner of all images
corners = images[:, :10, :10]
print(f"Top-left corners shape: {corners.shape}")  # (8, 10, 10)

# Get last 3 images, bottom-right 5x5 corner
subset = images[-3:, -5:, -5:]
print(f"Subset shape: {subset.shape}")  # (3, 5, 5)
