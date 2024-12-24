# Dimension Mixer: Group Mixing of Input Dimensions for Efficient Function Approximation
Arxiv: [Dimension Mixer](https://arxiv.org/abs/2311.18735)  

## LRA Benchmark
Adapted from [Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention](https://github.com/mlpen/Nystromformer)

## Setup

#### Requirements

Setup conda environment from environment.yml file at root directiory.   
`conda env create -f environment.yml`   
`conda activate py3`

*Alternatively*:   
This program uses python-3. Install required libraries.   
`pip install -r requirements.txt`


#### Dataset
To prepare the datasets, one would need to download the source code from [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena` folder in folder `LRA/datasets/` and also download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder `LRA/datasets/`. 

Exact script:
```
git clone https://github.com/google-research/long-range-arena
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar -xf lra_release.gz
```

The directory structure would be
```
datasets/long-range-arena
datasets/lra_release
```
Then, run `sh create_datasets.sh` from `LRA/datasets/` and it will create train, dev, and test dataset pickle files for each task.

*The dataset pathfinder.py processes 32, 64 and 128: Modify(uncomment) code to include pathfinder 256*

### Experiment

To run a LRA experiment, run the following command in `LRA/code` folder
```
python3 run_tasks.py --model <model> --task <task> --identifier <manual model ID for experiments & logs>
```
where `<model>` can be set to `softmax, nystrom-64, butterfly-64,reformer-2, performer-256, none ` corresponding to standard self-attention, Nystromformer with 64 landmarks, Butterfly Attention with 64 block size, Reformer with 2 LSHs, Performer with 256 random projection dimension and No-Attention. And `<task>` can be set to `listops, text, retrieval, image, pathfinder32-curv_contour_length_14, pathfinder128-curv_contour_length_14`. The best models and log files will be saved `LRA/logs/` folder.


**Other Hyperparameters:** See help (`python3 run_tasks.py --help`)
- Default hyperparameters are in `LRA/code/lra_config.py` file.


#### Our experiments:
We experiment butterfly attention for all Long Range Arena tasks.   
From `LRA/code` run:   
`sh run_tasks.sh`
#

## Solving-PathX
Solving the difficult problem of Pathfinder-X of Long Range Arena (LRA) Benchmark.   
* Follow instruction from [`./LRA/datasets/Generating-Pathfinder.md`](./LRA/datasets/Generating-Pathfinder.md) to generate pathfinder-128 dataset with increasing complexity.
### For Butterfly-Attention:

From `LRA/code/`   
1. First train on downloaded dataset `pathfinder32-curv_contour_length_14`, the script:   
`run_tasks.py --task pathfinder32-curv_contour_length_14 --model butterfly-32 --identifier butterfly_lr0.0003_s2023_pathfinder32-curv_contour_length_14 --batch_size 256 --gpu_batch_size 256 --seed_data 123 --seed_model 2023 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4`    
*You may change the* `--dataset_root` *and* `--log_dir` *arguments according to your requirements.*

2. Train on downloaded dataset `pathfinder64-curv_contour_length_9` with same configurations, and doubled butterfly-*block_size* -> initialized from previous experiment which is saved inside `../logs/` (by default):   
`run_tasks.py --task pathfinder64-curv_contour_length_9 --model butterfly-64 --identifier butterfly_lr0.0003_s2023_pathfinder64-curv_contour_length_9 --batch_size 64 --gpu_batch_size 64 --seed_data 456 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathfinder32-curv_contour_length_14_output.model`   
*Your previous* `--log_dir` *is the new* `--pretrained_root`.

3. Train on generated dataset `pathX-cl14_nogap` with same configurations, and doubled butterfly-*block_size* -> initialized from previous experiment:   
`run_tasks.py --task pathX-cl14_nogap --model butterfly-128 --identifier butterfly_lr0.0003_s2023_pathX-cl14_nogap --batch_size 64 --gpu_batch_size 32 --seed_data 789 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathfinder64-curv_contour_length_9_output.model`

4. Train on generated dataset `pathX-cl14_alpha0.0` with same configurations, initialized from previous experiment:   
`run_tasks.py --task pathX-cl14_alpha0.0 --model butterfly-128 --identifier butterfly_lr0.0003_s2023_pathX-cl14_alpha0.0 --batch_size 64 --gpu_batch_size 32 --seed_data 147 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathX-cl14_nogap_output.model`

5. Train on generated dataset `pathX-cl14_alpha0.75` with same configurations, initialized from previous experiment:   
`run_tasks.py --task pathX-cl14_alpha0.75 --model butterfly-128 --identifier butterfly_lr0.0003_s2023_pathX-cl14_alpha0.75 --batch_size 64 --gpu_batch_size 32 --seed_data 258 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathX-cl14_alpha0.0_output.model`

6. Train on generated dataset `pathX-cl14_alpha1.25` with same configurations, initialized from previous experiment:   
`run_tasks.py --task pathX-cl14_alpha1.25 --model butterfly-128 --identifier butterfly_lr0.0003_s2023_pathX-cl14_alpha1.25 --batch_size 64 --gpu_batch_size 32 --seed_data 369 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathX-cl14_alpha0.75_output.model`

7. Train on downloaded dataset `pathfinder128-curv_contour_length_14` with same configurations, initialized from previous experiment:   
`run_tasks.py --task pathfinder128-curv_contour_length_14 --model butterfly-128 --identifier butterfly_lr0.0003_s2023_pathfinder128-curv_contour_length_14 --batch_size 64 --gpu_batch_size 32 --seed_data 321 --warmup 500 --num_train_steps 25000 --lr_decay cos --weight_decay 0.001 --learning_rate 0.0003 --attention_dropout 0.1 --embedding_dropout 0.1 --mlp_dropout 0.1 --dropout_prob 0.0 --num_head 4 --head_dim 16 --num_layers 4 --pretrained_root ../logs --pretrained_init butterfly_lr0.0003_s2023_pathX-cl14_alpha1.25_output.model`

### For Other Attention:

Follow the same script above; for example Nystromformer, replace the `--model` parameter `butterfly-N` with `nystrom-N` and follow the exact steps from 1-6.

```
@article{sapkota2023dimension,
  title={Dimension Mixer: Group Mixing of Input Dimensions for Efficient Function Approximation},
  author={Sapkota, Suman and Bhattarai, Binod},
  journal={arXiv preprint arXiv:2311.18735},
  year={2023},
  url={https://arxiv.org/abs/2311.18735}
}
```
