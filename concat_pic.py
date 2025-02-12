import os
import torch
import numpy as np
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
from torchvision import transforms

root_path = "../textual_inversion_back_/outputs/txt2img-sample/diversity/n02687172/samples"
img_list = ["1995.jpg", "2097.jpg", "0439"]
# img_list = ["0439.jpg", "0444.jpg", "1643.jpg", "1645.jpg", "1995.jpg", "2097"]
all_samples=list()
t = transforms.ToTensor()
for _, img_name in enumerate(img_list):
    im = Image.open(os.path.join(root_path, img_name)).convert('RGB')
    img = t(im)
    img = img.unsqueeze(0)
    all_samples.append(img)
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=3)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
# grid = grid.cpu().numpy()
Image.fromarray(grid.astype(np.uint8)).save("./aircraft_carrier.png")
print("finish")
