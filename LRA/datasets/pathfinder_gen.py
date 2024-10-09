
import numpy as np
import os
import pickle
import tensorflow as tf
import random

import sys

SEED = 2222
random.seed(SEED)

data_dir = str(sys.argv[1]).strip()

### Get the name of the folder currently processing
subdir = data_dir.split("/")[-1]
if data_dir.endswith("/"):
    subdir = data_dir.split("/")[-2]

metadata_list = [
    os.path.join(data_dir, "metadata", file)
    for file in os.listdir(os.path.join(data_dir, "metadata"))
    if file.endswith(".npy") and ("bkp" not in os.path.basename(file))
]

print(metadata_list)

ds_list = []
for idx, metadata_file in enumerate(metadata_list):
    print(idx, len(metadata_list), metadata_file, "\t\t", end = "\r")
    for inst_meta in np.load(metadata_file):
        metadata = [d.decode() for d in inst_meta]
        img_path = os.path.join(data_dir, metadata[0], metadata[1])
        img_bin = tf.io.read_file(img_path)
        if len(img_bin.numpy()) == 0:
            print()
            print("detected empty image")
            continue
        img = tf.image.decode_png(img_bin)
        seq = img.numpy().reshape(-1).astype(np.int32)
        label = int(metadata[3])
        ds_list.append({"input_ids_0":seq, "label":label})

random.shuffle(ds_list)

bp80 = int(len(ds_list) * 0.8)
bp90 = int(len(ds_list) * 0.9)
train = ds_list[:bp80]
dev = ds_list[bp80:bp90]
test = ds_list[bp90:]


# root = sys.argv[1]
root = "./"
path = os.path.join(root, f"pathX-{subdir}.train.pickle")
print('dumping ', path)
with open(path, "wb") as f:
    pickle.dump(train, f)

path = os.path.join(root, f"pathX-{subdir}.dev.pickle")
print('dumping ', path)
with open(path, "wb") as f:
    pickle.dump(dev, f)

path = os.path.join(root, f"pathX-{subdir}.test.pickle")
print('dumping ', path)
with open(path, "wb") as f:
    pickle.dump(test, f)
