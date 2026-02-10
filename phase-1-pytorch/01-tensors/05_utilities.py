import torch

# ============================================
# CONCATENATION: Joining Tensors Together
# ============================================

# Scenario: You have two batches of data, want to combine them

batch1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

batch2 = torch.tensor([[7, 8, 9], [10, 11, 12]])  # Shape: (2, 3)

# Concatenate along rows (dim = 0) - "Stack more samples"
combined_rows = torch.cat([batch1, batch2], dim=0)
print(combined_rows)
print(combined_rows.shape)

# Concatenate along columns (dim = 1) - "Add more features"
combined_cols = torch.cat([batch1, batch2], dim=1)
print(combined_cols)
print(combined_cols.shape)

# ============================================
# STACKING: Creates a NEW Dimension
# ============================================

# Scenario: You have 3 separate images, want to create a batch

img1 = torch.rand(28, 28)  # Shape: (28, 28)
img2 = torch.rand(28, 28)  # Shape: (28, 28)
img3 = torch.rand(28, 28)  # Shape: (28, 28)

# Stacj creates a NEW dimension
batch = torch.stack([img1, img2, img3], dim=0)
print(batch.shape)  # (3, 28, 28)

# Compare with cat — cat would FAIL here because no batch dim exists!
# torch.cat([img1, img2], dim=0) would give (56, 28) — not what we want


# ============================================
# MIN, MAX, SUM, MEAN — Aggregation Operations
# ============================================


# Scenario: Exam scores — 4 students, 3 exams each ( students are rows , exams are columns )
scores = torch.tensor(
    [[85.0, 90.0, 78.0], [92.0, 88.0, 95.0], [70.0, 75.0, 80.0], [88.0, 82.0, 91.0]]
)

print(scores)

# Global opeations (entire tensor)
print("Max: ", scores.max())
print("Min: ", scores.min())
print("Sum: ", scores.sum())
print("Global mean: ", scores.mean())

# Along specific dimension
# dim = 0: operate across rows (result per column/exam)

print("Max per exam: ", scores.max(dim=0).values)
print("Mean per exam: ", scores.mean(dim=0))

# dim = 0: operate across rows (result per column/exam)

print("Max per exam: ", scores.max(dim=1).values)
print("Mean per exam: ", scores.mean(dim=1))

# ============================================
# ARGMAX / ARGMIN — Finding the INDEX of max/min
# ============================================

# Scenario: Which exam did each student score highest on?

scores = torch.tensor(
    [
        [85.0, 90.0, 78.0],  # Student 0: best at exam 1
        [92.0, 88.0, 95.0],  # Student 1: best at exam 2
        [70.0, 75.0, 80.0],  # Student 2: best at exam 2
        [88.0, 82.0, 91.0],
    ]
)

best_exam_per_student = scores.argmax(dim=1)
print("Best exam index per student: ", best_exam_per_student)  # [1 ,2 , 2 , 2]

# Scenario: Which student scored highest on each exam?
best_student_per_exam = scores.argmax(dim=0)
print(best_student_per_exam)  # [1, 0, 1]

# ============================================
# USEFUL CREATION UTILITIES
# ============================================

# arange: Like python's range()
sequence = torch.arange(0, 10, 2)
print(sequence)

# linspace: evenly spaced values
smooth = torch.linspace(0, 1, 5)  # start, end, num_points
print(smooth)

# zeroes_like / ones_like: same shape as another tensor

template = torch.rand(3, 4)
zeroes_copy = torch.zeros_like(template)
print(template.shape)
print(zeroes_copy.shape)

"""
The Logic zeroes_like(x): Returns a tensor of the same shape/type as $x$, filled with 0.ones_like(x): Returns a tensor of the same shape/type as $x$, filled with 1.
"""
