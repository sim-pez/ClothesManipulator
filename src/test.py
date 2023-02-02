
#%%
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
im1=os.path.join(par.ROOT_DIR,"dati/Images/Female/4_Shirts_blouses/0045884_16.jpg")
# read images from computer
a = read_image(im1)
b = read_image(im1)
c = read_image(im1)
d = read_image(im1)
  
# make grid from the input images
# this grid contain 4 columns and 1 row
Grid = make_grid([a, b, c, d])
  
# display result
img = torchvision.transforms.ToPILImage()(Grid)
img.show()
# %%
