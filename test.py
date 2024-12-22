from torch.nn import CrossEntropyLoss
import torch
loss = CrossEntropyLoss(reduction="none")
input = torch.randn(10, 5, 4, requires_grad=True)
target = torch.empty(10, 5, dtype=torch.long).random_(5)
print("input=", input)
print("target=", target)
output = loss(input, target)
print("output=", output)