import os, sys
import random

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torchvision.io import read_image

## load data 
data_dir = str(sys.argv[1]).strip()
num_images = int(sys.argv[2])


imgs = os.path.join(data_dir, "imgs")
folders = os.listdir(imgs)
folders = [folder for folder in folders if "bkp" not in folder ]
random.shuffle(folders)

imgs_list = []
for folder in folders:
    imgs = os.listdir(os.path.join(data_dir, "imgs", folder))
    imgs_list += [os.path.join(data_dir, "imgs", folder, img) for img in imgs if img.endswith(".png")]

random.shuffle(imgs_list)
selected_imgs = imgs_list[:num_images]

grid = make_grid([read_image(im) for im in selected_imgs], nrow=num_images, pad_value=255)
img = ToPILImage()(grid)

## To save the image.
### Get the name of the folder currently processing
subdir = data_dir.split("/")[-1]
if data_dir.endswith("/"):
    subdir = data_dir.split("/")[-2]
os.makedirs("processed", exist_ok=True)
img.save("processed/pathX-"+subdir+".png")