
# %%
import os
import sys
from pathlib import Path

notebook_path = Path().absolute()
sys.path.append(str(notebook_path.parent))

# %%
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
from utils import newton_dataset, newton_dataset_new, emotion_dataset

SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# %%


# %%
custom_cache_dir = "/scratch/bbjr/skarmakar/huggingface"

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name='llama_3_8b_it'

language_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    cache_dir=custom_cache_dir,
)

use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    use_fast=use_fast_tokenizer, 
    padding_side="left", 
    legacy=False,
)

# tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %%
controller = NeuralController(
    language_model,
    tokenizer,
    rfm_iters=8,
    batch_size=4,
    control_method='rfm'
)

# %%
concept_types = ["Cam", "Isaac"]
data_dir = "../data/prefixed_newton"
dataset = newton_dataset_new(data_dir, controller, concept_types, samples=350, seed=SEED)

# concept_types = ["love", "hate"]
# data_dir = "../data/emotions"
# dataset = emotion_dataset(data_dir, controller, concept_types, samples=290)

# %%
# print(len(dataset["love"]["train"]["inputs"]))
# print(len(dataset["love"]["train"]["labels"]))

# print(dataset["love"]["train"]["inputs"][0])
# print(dataset["love"]["train"]["inputs"][1])
# print(dataset["love"]["train"]["labels"][0])
# print(dataset["love"]["train"]["inputs"][2])
# print(dataset["love"]["train"]["inputs"][3])
# print(dataset["love"]["train"]["labels"][1])

# %%


# %%


# %%
def find_lingering_forward_hooks(model):
    lingering_hooks = []
    for name, module in model.named_modules():
        if module._forward_hooks:
            hook_info = (name, 'forward')
            lingering_hooks.append(hook_info)
            print(f"Warning: Found {len(module._forward_hooks)} lingering 'forward' hook(s) on module: {name}")

        if module._forward_pre_hooks:
            hook_info = (name, 'forward_pre')
            lingering_hooks.append(hook_info)
            print(f"Warning: Found {len(module._forward_pre_hooks)} lingering 'forward_pre' hook(s) on module: {name}")
            
    if not lingering_hooks:
        print("Success: No lingering forward hooks found in the model.")
        
    return lingering_hooks

# %%
# rfm_iters = 16
rfm_iters = 8
# batch_size = 8
batch_size = 4
n_components = 300
# n_components = 5
# energy = 0.98

# %%
# controllers = {}

# for concept_type in tqdm(concept_types):
    
#     other_type = [k for k in concept_types if k != concept_type][0]
    
#     train_data = dataset[concept_type]['train']
#     test_data = dataset[concept_type]['test']
    
#     controller = NeuralController(
#         language_model,
#         tokenizer,
#         rfm_iters=rfm_iters,
#         batch_size=batch_size,
#         control_method='rfm',
#         n_components=n_components,
#         # energy=energy,
#     )
    
#     controller.compute_directions(train_data['inputs'], train_data['labels'])
    
#     controllers[concept_type] = controller

# %%
component_idx = 1
t_ite = 15

# anti = "yes"
anti = "no"

# control_coef=1.0
# control_coef=0.4
control_coef=0.9

base_path = "../directions/stable_base_config"
# path = f"{base_path}/isaac_cam_{n_components}_orig_ite_d{component_idx}_{SEED}"
path = f"{base_path}/isaac_cam_{n_components}_orig_ite_c{control_coef}_{SEED}"

# %%
# controllers = {}

print("Starting training:")

for concept_type in tqdm(concept_types):
    
    other_type = [k for k in concept_types if k != concept_type][0]
    
    train_data = dataset[concept_type]['train']
    test_data = dataset[concept_type]['test']
    
    controller = NeuralController(
        language_model,
        tokenizer,
        rfm_iters=rfm_iters,
        batch_size=batch_size,
        control_method='rfm',
        n_components=n_components,
        # energy=energy,
    )

    # hidden_layers = None
    # hidden_layers = [-1, -2, -3, -4, -5]

    
    # controller.compute_directions(train_data['inputs'], train_data['labels'])

    controller.compute_directions_ite(
        train_data['inputs'], 
        train_data['labels'],
        # hidden_layers=hidden_layers,
        control_coef=control_coef,
        component_idx=component_idx,
        anti=anti,
    )

    # controller.compute_directions_ite_all(
    #     train_data['inputs'], 
    #     train_data['labels'],
    #     # hidden_layers=hidden_layers,
    #     control_coef=1.0,
    #     component_idx=component_idx,
    #     t_ite=t_ite,
    # )
    
    # controllers[concept_type] = controller



    # base_path = "/scratch/bbjr/skarmakar/dementia/ckpt"
    # base_path = "../directions/stable"
    # path = f"{base_path}/isaac_cam_{n_components}_itea_d{component_idx}_{SEED}"
    
    os.makedirs(path, exist_ok=True)
    controller.save(concept=f'{concept_type}', model_name='llama_3_8b_it', path=path)
    # break

# %%
lh = find_lingering_forward_hooks(language_model)

# %%
# print(controllers["Cam"].directions.keys())
# print(len(controllers["Cam"].directions[-1]))
# print(controllers["Cam"].directions[-1][0].shape)

# %%


# %%
# /u/skarmakar1/miniconda3/envs/neucon/lib/python3.10/site-packages/xrfm/rfm_src/recursive_feature_machine.py

# %%
# print(controllers["Cam"].directions.keys())
# print(controllers["Cam"].directions[-31].shape)
# print(controllers["Isaac"].directions.keys())
# print(controllers["Isaac"].directions[-31].shape)

# %%


# %%


# %%


# %%


# %%
# base_path = "/scratch/bbjr/skarmakar/dementia/ckpt"
# # base_path = "../directions/stable"

# # path = f"../directions/stable/isaac_cam_{n_components}_ite"
# # path = f"../directions/stable/isaac_cam_{n_components}_3_{SEED}"
# # path = f"../directions/stable/emotions_{n_components}_2"

# # path = f"../directions/stable/isaac_cam_{n_components}_itea_d{component_idx}_{SEED}"

# path = f"{base_path}/isaac_cam_{n_components}_itea_d{component_idx}_{SEED}"

# os.makedirs(path, exist_ok=True)

# for concept_type in concept_types:
#     controller = controllers[concept_type]
#     # other_type = [k for k in concept_types if k!=concept_type][0]
    
#     controller.save(concept=f'{concept_type}', model_name='llama_3_8b_it', path=path)

# %%


# %%



