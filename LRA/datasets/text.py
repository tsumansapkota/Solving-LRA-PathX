import sys
sys.path.append("./long-range-arena/lra_benchmarks/text_classification/")
import input_pipeline
import numpy as np
import pickle

import tensorflow as tf
import random

SEED = 2222
random.seed(SEED)
np.random.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
tf.random.set_seed(SEED)

seq_len = 4096 #1024
max_length = 4000 #1000
remain = seq_len-max_length

train_ds, eval_ds, test_ds, encoder = input_pipeline.get_tc_datasets(
    n_devices = 1, task_name = "imdb_reviews", data_dir = None,
    batch_size = 1, fixed_vocab = None, max_length = max_length)

mapping = {"train":train_ds, "dev": eval_ds, "test":test_ds}
for component in mapping:
    ds_list = []
    for idx, inst in enumerate(iter(mapping[component])):
        ds_list.append({
            "input_ids_0":np.concatenate([inst["inputs"].numpy()[0], np.zeros(remain, dtype = np.int32)]),
            "label":inst["targets"].numpy()[0]
        })
        if idx % 100 == 0:
            print(f"{idx}\t\t", end = "\r")
    with open(f"text.{component}.pickle", "wb") as f:
        pickle.dump(ds_list, f)
