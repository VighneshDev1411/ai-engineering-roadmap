import torch

# ============================================
# RESHAPING BASICS
# ============================================

# Create a tensor with 12 elements 
orignal = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(f"Original: {orignal}")
print(f"Orignial shape: {orignal.shape}") #(12, )

print("\n=== Different Reshapes ===")

reshaped_3x4 = orignal.reshape(3, 4)
print(f"(3, 4):\n{reshaped_3x4}")

# Reshape to 2D: 4 rows, 3 columns
reshaped_4x3 = orignal.reshape(4, 3)
print(f"\n(4, 3):\n{reshaped_4x3}")

# Reshape to 2D: 2 rows, 6 columns
reshaped_2x6 = orignal.reshape(2, 6)
print(f"\n(2, 6):\n{reshaped_2x6}")

# Reshape to 3D: 2 blocks, 2 rows, 3 columns
reshaped_3d = orignal.reshape(2, 2, 3)
print(f"\n(2, 2, 3):\n{reshaped_3d}")

# ============================================
# THE -1 TRICK: Auto-calculate dimension
# ============================================

data = torch.rand(1000, 28, 28) # 1000 images, 28x28 each
print(f"\nOrignal images shape: {data.shape}")

# Flatten each image: keep 1000 samples, auto-calculate the rest
flattened = data.reshape(1000, -1) # -1 means "figure it out"
print(f"Flattened shape: {flattened.shape}") # (1000, 784)

# PyTorch calculated: 28 x 28 = 784

# Another example: you know you want 4 colums, auto-calculate rows
x = torch.arange(12) # [0, 1, 2, ...., 11]
reshaped = x.reshape(-1, 4) # "however many rows needed, 4 columns"
print(f"\nAuto rows:\n{reshaped}")  # (3, 4)

# Common Reshaping Operations
# ============================================
# COMMON PATTERNS IN DEEP LEARNING
# ============================================

# Pattern 1: Flatten (for feeding into linear layers)
images = torch.rand(32, 28, 28) # 32 Grayscale images
flat = images.reshape(32, -1) # (32, 784)
print(f"Flattened: {flat.shape}")

# Pattern 2: Add batch dimension (single image -> batch of 1)
single_image = torch.rand(28, 28) # One Image
batched = single_image.reshape(1, 28, 28) # Batch of 1
print(f"Batched single image: {batched.shape}")

# Pattern 3: Add channel dimension (grayscale -> CNN format)
# CNNs expect: (batch, channels, height, width)
grayscale_batch = torch.rand(32, 28, 28) # (batch, H, W)
cnn_format = grayscale_batch.reshape(32, 1, 28, 28) # (batch, C, H, W)
print(f"CNN format: {cnn_format.shape}")

x = torch.arange(12)
a = x.view(3, 4)
b = x.reshape(3,4)
print(f"view: {a.shape}, reshape: {b.shape}")

# Real Scenario : Image Classification 

# Step 1: Load batch of images (from data loader)
batch_images = torch.rand(32, 28 ,28) # 32 grayscale images
print(f"Loaded batch: {batch_images.shape}")

# Step 2: Reshape for CNN (add channel dimension)
# CNN expects: (batch, channels, height, width)

cnn_input = batch_images.reshape(32, 1, 28, 28)
print(f"CNN input: {cnn_input.shape}")

# ... CNN processes it ...
# Let's say CNN outputs: (32, 64, 7, 7) → 32 samples, 64 feature maps, 7x7 each

cnn_output = torch.rand(32, 64, 7, 7)
print(f"CNN output: {cnn_output.shape}")

# Step 3: Flatten for fully connected layer
# FC layer expects: (batch, features)
fc_input = cnn_output.reshape(32, -1) # 64 x 7 x 7 = 3136 Automatic calcuation by pytorch
print(f"FC input: {fc_input.shape}") 




"""
Notes: 

Lines 73-94: Image Classification Workflow


  This section simulates how tensors are transformed as they pass through different stages of a Convolutional Neural Network (CNN).


   1. Step 1: Loading a Batch (Lines 76-78)
        batch_images = torch.rand(32, 28, 28) # 32 grayscale images
       - Here, you have a batch of 32 images, each $28 \times 28$ pixels (like MNIST).
       - At this stage, the tensor shape is [batch_size, height, width].


   2. Step 2: Reshaping for CNN (Lines 80-84)
        cnn_input = batch_images.reshape(32, 1, 28, 28)
       - Standard PyTorch CNN layers (Conv2d) expect a 4D tensor: (Batch, Channels, Height, Width).
       - Since these are grayscale, you explicitly add a "1" for the channel dimension. If they were RGB, this would be "3".


   3. Intermediate State (Lines 86-90)
       - This part simulates the output of a CNN. After several convolutions, the spatial size typically shrinks (from $28 \times 28$ to $7 \times 7$), but the number of
         channels (feature maps) increases (from 1 to 64).


   4. Step 3: Flattening for Fully Connected Layer (Lines 92-94)
        fc_input = cnn_output.reshape(32, -1)
       - Before passing data into a standard linear (Dense/FC) layer, you must "flatten" the multi-dimensional feature maps into a single vector per image.
       - The `-1` trick: PyTorch automatically calculates the missing dimension. In this case, $64 \times 7 \times 7 = 3136$, so the resulting shape becomes [32, 3136].
"""