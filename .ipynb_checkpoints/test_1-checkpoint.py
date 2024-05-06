import torch
import torch.nn.functional as F

# Example inputs
batch_size = 4
num_classes = 3

# Mock model predictions (logits before softmax)
y_pred = torch.randn(batch_size, num_classes)  # Example logits
labels = torch.tensor([1, 2, 1, 1])# Example target labels


print(y_pred)

# Calculate cross entropy loss
loss = F.cross_entropy(y_pred, labels)

print(loss)