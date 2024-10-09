import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import lra_config

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model", dest="model", required=True)
parser.add_argument("--task", type=str, help="task", dest="task", required=True)
parser.add_argument(
    "--skip_train", type=int, help="skip_train", dest="skip_train", default=0
)
parser.add_argument(
    "--wandb_mode",
    help="Mode for Wandb: modes can be: online, offline, disabled",
    type=str,
    # default="online",
    default="disabled",
)
parser.add_argument(
    "--identifier",
    type=str,
    help="Training ID to identify each training seperately (same model could be run with different ID for differentiating",
    default="",
)

### Additional Arguments
parser.add_argument(
    "--pretrained_init",
    help="Initialize with pretrained weight of same model",
    type=str,
    default=None,
)
parser.add_argument(
    "--pretrained_root",
    help="directory for pretrained weight of model",
    type=str,
    default=None,
)
parser.add_argument(
    "--dataset_root",
    help="datasets root",
    type=str,
    default="../datasets",
)
##### EXTRA PARAMETERS: INITIALIZATIONS
parser.add_argument(
    "--seed_model", help="Seed for model; default is None", default=-1, type=int
)
parser.add_argument(
    "--seed_data", help="Seed for data loader; default is None", default=-1, type=int
)

### Transformer Parameters
parser.add_argument("--num_layers", type=int, help="number of layers", default=None)
parser.add_argument(
    "--embedding_dim", default=None, help="model dimension for encoder", type=int
)
parser.add_argument(
    "--transformer_dim", default=None, help="model dimension for transformer", type=int
)
parser.add_argument(
    "--transformer_hidden_dim",
    help="number of hidden neurons in MLP layer",
    default=None,
    type=int,
)
parser.add_argument(
    "--num_head", type=int, help="number of attention heads", default=None
)
parser.add_argument(
    "--head_dim", default=None, help="dimension of each attention heads", type=int
)
parser.add_argument(
    "--vocab_size", help="Vocabulary size of input values", default=None, type=int
)
parser.add_argument("--attention_grad_checkpointing", default=None, action="store_true")
parser.add_argument(
    "--dropout_prob",
    help="dropout for embeddings, MLP and attention before residual",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--embedding_dropout",
    help="dropout for embeddings; default override by dropout_prob",
    default=None,
    type=float,
)
parser.add_argument(
    "--mlp_dropout",
    help="dropout for MLP; default override by dropout_prob",
    default=None,
    type=float,
)
parser.add_argument(
    "--attention_dropout",
    type=float,
    help="attention dropout",
    dest="attention_dropout",
    default=None,
)

### Training Parameters
parser.add_argument(
    "--learning_rate",
    type=float,
    help="learning rate of model",
    dest="learning_rate",
    default=None,
)
parser.add_argument("--batch_size", default=None, type=int)
parser.add_argument("--warmup", help="Warmup steps", default=None, type=int)
parser.add_argument(
    "--lr_decay",
    help="anneal_strategy for torch.optim.lr_scheduler.OneCycleLR",
    default=None,
    type=str,
)
parser.add_argument("--weight_decay", default=None, type=float)
parser.add_argument(
    "--eval_frequency", help="Evaluate every N steps", default=None, type=int
)
parser.add_argument(
    "--num_train_steps", help="Number of total training steps", default=None, type=int
)
parser.add_argument(
    "--num_eval_steps",
    help="Numbr of total evaluatation steps per one evaluation",
    default=None,
    type=int,
)

### GPU configuration
parser.add_argument("--gpu_batch_size", default=None, type=int)

### Log directiory
parser.add_argument("--log_dir", help="Log directory", type=str, default="../logs/")

args = parser.parse_args()

attn_type = args.model
task = args.task
## renamed to "pathfinder128-curv_contour_length_14" for model config
if task.startswith("pathX"):
    task = "pathfinder128-curv_contour_length_14"

checkpoint_dir = args.log_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

print(lra_config.config[task]["extra_attn_config"].keys(), flush=True)


### New configs
def modify_model_config(args, task):
    model_config = lra_config.config[task]["model"]

    ### Custom setting from argument | Override lra_config.py
    ####################################################
    if args.num_layers:
        model_config["num_layers"] = args.num_layers
    if args.embedding_dim:
        model_config["embedding_dim"] = args.embedding_dim
    if args.transformer_dim:
        model_config["transformer_dim"] = args.transformer_dim
    if args.transformer_hidden_dim:
        model_config["transformer_hidden_dim"] = args.transformer_hidden_dim
    if args.num_head:
        model_config["num_head"] = args.num_head
    if args.head_dim:
        model_config["head_dim"] = args.head_dim
    if args.vocab_size:
        model_config["vocab_size"] = args.vocab_size

    if args.attention_grad_checkpointing:
        model_config["attention_grad_checkpointing"] = args.attention_grad_checkpointing
    if args.attention_dropout is not None:
        model_config["attention_dropout"] = args.attention_dropout

    ####################################################
    ### Dropouts
    if args.dropout_prob is None:
        args.dropout_prob = model_config["dropout_prob"]
    else:
        model_config["dropout_prob"] = args.dropout_prob
    if args.embedding_dropout is None:
        model_config["embedding_dropout"] = args.dropout_prob
    else:
        model_config["embedding_dropout"] = args.embedding_dropout
    if args.mlp_dropout is None:
        model_config["mlp_dropout"] = args.dropout_prob
    else:
        model_config["mlp_dropout"] = args.mlp_dropout

    ####################################################

    model_config["seed_model"] = args.seed_model
    model_config["pretrained_init"] = args.pretrained_init
    if (args.pretrained_root is not None) and (args.pretrained_init is not None):
        model_config["pretrained_init"] = os.path.join(
            args.pretrained_root, args.pretrained_init
        )

    ####################################################
    return model_config


model_config = modify_model_config(args, task)
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["max_seq_len"] = int(
    2 ** math.ceil(math.log2(model_config["max_seq_len"]))
)


####################################################
def modify_training_config(args, task):
    ###### Modify default training config
    training_config = lra_config.config[task]["training"]

    ### Custom setting from argument | Override lra_config.py
    if args.batch_size:
        training_config["batch_size"] = args.batch_size
    if args.learning_rate:
        training_config["learning_rate"] = args.learning_rate
    if args.warmup:
        training_config["warmup"] = args.warmup
    if args.lr_decay:
        training_config["lr_decay"] = args.lr_decay
    if args.weight_decay:
        training_config["weight_decay"] = args.weight_decay

    if args.eval_frequency:
        training_config["eval_frequency"] = args.eval_frequency
    if args.num_train_steps:
        training_config["num_train_steps"] = args.num_train_steps
    if args.num_eval_steps:
        training_config["num_eval_steps"] = args.num_eval_steps

    return training_config


training_config = modify_training_config(args, task)

gpu_memory = lra_config.config[task]["gpu_memory"][attn_type]
if args.gpu_batch_size is not None:
    gpu_memory = args.gpu_batch_size

### For pathX_cl14_alpha{} and nogap , the task is not supported hence renamed to "pathfinder128-curv_baseline" for model config
task = args.task

identifier = f"{task}_{attn_type}"
if len(args.identifier) > 0:
    identifier = args.identifier


device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent=4))

####################################################
wandb.init(
    ## modes can be "online", "offline", "disabled"
    mode=args.wandb_mode,
    project="LRA-butterfly",
    entity="dimension-mixer",
    name=identifier,
    config={
        "ID": identifier,
        "attention": attn_type,
        "task": task,
        "model_config": model_config,
        "training_config": training_config,
        "gpu_memory": gpu_memory,
    },
)

if task == "retrieval":
    model = ModelForSCDual(model_config)
else:
    model = ModelForSC(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
print(
    f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}",
    flush=True,
)

model = model.cuda()
# model = torch.compile(model)
model = nn.DataParallel(model, device_ids=device_ids)

ds_iter = {
    "train": enumerate(
        DataLoader(
            LRADataset(
                os.path.join(args.dataset_root, f"{task}.train.pickle"),
                True,
                args.seed_data,
            ),
            batch_size=training_config["batch_size"],
            drop_last=True,
        )
    ),
    "dev": enumerate(
        DataLoader(
            LRADataset(
                os.path.join(args.dataset_root, f"{task}.dev.pickle"),
                True,
                args.seed_data,
            ),
            batch_size=training_config["batch_size"],
            drop_last=True,
        )
    ),
    "test": enumerate(
        DataLoader(
            LRADataset(
                os.path.join(args.dataset_root, f"{task}.test.pickle"),
                False,
                args.seed_data,
            ),
            batch_size=training_config["batch_size"],
            drop_last=True,
        )
    ),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config["learning_rate"],
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=training_config["weight_decay"],
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=training_config["learning_rate"],
    pct_start=training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy=training_config["lr_decay"],
    total_steps=training_config["num_train_steps"],
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None


def step(component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim=0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim=0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(
        f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t",
        end="\r",
        flush=True,
    )

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)


def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        ## wandb summary
        wandb.run.summary["best_accuracy"] = summary["best_accu"]

        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save(
                {"model_state_dict": model.module.state_dict()},
                log_f_path.replace(".log", ".model"),
            )
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx": train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush=True)
    log_f.write(json.dumps(summary_round, sort_keys=True) + "\n")
    log_f.flush()

    #### Save to wandb
    prefix = summary_round["component"]
    new_summary = {}
    for key in summary_round:
        if key == "component":
            continue
        new_summary[f"{prefix}-{key}"] = summary_round[key]
    wandb.log(new_summary)

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []


init_t = time.time()

log_f_path = os.path.join(checkpoint_dir, f"{identifier}_output.log")
log_f = open(log_f_path, "a+")

summary = {
    component: {"t": 0, "loss": [], "accu": [], "best_accu": 0, "component": component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] / len(device_ids) / gpu_memory, 1)

accumu_steps = max(training_config["batch_size"] / len(device_ids) / gpu_memory, 1)
accumu_steps = int(np.ceil(accumu_steps))
print(f"accumu_steps={accumu_steps}")

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config["num_train_steps"]):
            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx)
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location="cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print_summary(summary["test"], False, train_step_idx)
