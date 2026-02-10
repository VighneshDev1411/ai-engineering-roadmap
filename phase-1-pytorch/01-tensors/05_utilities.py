import torch

# ============================================
# CONCATENATION: Joining Tensors Together
# ============================================

# Scenario: You have two batches of data, want to combine them

batch1 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]]) # Shape: (2, 3)

batch2 = torch.tensor([[7, 8, 9],
                       [10, 11, 12]]) # Shape: (2, 3)

# Concatenate along rows (dim = 0) - "Stack more samples"
combined_rows = torch.cat([batch1, batch2], dim=0)
print(combined_rows)
print(combined_rows.shape)

# Concatenate along columns (dim = 1) - "Add more features"
combined_cols = torch.cat([batch1, batch2], dim=1)
print(combined_cols)
print(combined_cols.shape)
