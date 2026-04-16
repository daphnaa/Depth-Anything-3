import torch
from depth_anything_3.api import DepthAnything3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("DA3-BASE").to(device)

print(type(model))
print("has model:", hasattr(model, "model"))
if hasattr(model, "model"):
    print("inner type:", type(model.model))