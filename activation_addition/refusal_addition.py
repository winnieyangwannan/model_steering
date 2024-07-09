
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
import numpy as np
import argparse

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', default="Qwen/Qwen-1_8B", type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--N_INST_TRAIN', default=32, type=int)
    parser.add_argument('--N_INST_TEST', default=32, type=int)
    parser.add_argument('--pos', default=-1, type=int)
    parser.add_argument('--extract_layer', default=23, type=int)
    parser.add_argument('--strength', default=10, type=int)
    parser.add_argument('--save_path', default= '/gpfs/data/buzsakilab/wy547/Code/model_steering/activation_addition', type=str)

    return parser.parse_args()

DEVICE = 'cuda'
args = parse_args()
MODEL_PATH = args.MODEL_PATH
model_name = args.MODEL_PATH.split('/')[-1]
batch_size = args.batch_size
N_INST_TRAIN = args.N_INST_TRAIN
N_INST_TEST = args.N_INST_TEST
pos = args.pos
extract_layer = args.extract_layer
strength = args.strength
save_path = args.save_path

print(f"model: {model_name}")
print(f"batch_size: {args.batch_size}")
print(f"N_INST_TRAIN: {args.N_INST_TRAIN}")
print(f"N_INST_TEST: {args.N_INST_TEST}")
print(f"pos: {args.pos}")
print(f"extract_layer: {args.extract_layer}")
print(f"strength: {args.strength}")
print(f"save_path: {args.save_path}")



#%% 1. Load Model

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=torch.float16,
    default_padding_side='left',
    fp16=True
)

model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = '<|extra_0|>'

#%% 2. Load Instruction

def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test

def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()



print("Harmful instructions:")
for i in range(4):
    print(f"\t{repr(harmful_inst_train[i])}")
print("Harmless instructions:")
for i in range(4):
    print(f"\t{repr(harmless_inst_train[i])}")


#%% 3. Tokenization

# QWEN_CHAT_TEMPLATE = """<|im_start|>user
# {instruction}<|im_end|>
# <|im_start|>assistant
# """

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    # prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
    prompts = [instruction for instruction in instructions]

    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)

# tokenize instructions
harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN])
harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:N_INST_TRAIN])

#%% 4. Run with Hook
# run model on harmful and harmless instructions, caching intermediate activations
harmful_logits, harmful_cache = model.run_with_cache(harmful_toks, names_filter=lambda hook_name: 'resid' in hook_name)
harmless_logits, harmless_cache = model.run_with_cache(harmless_toks, names_filter=lambda hook_name: 'resid' in hook_name)




#  PCA on activation


label_text = []
label = []
for ii in range(len(harmful_cache['resid_pre', extract_layer][:, pos, :])):
    label_text = np.append(label_text, 'harmful')
    label = np.append(label, 0)
for ii in range(len(harmless_cache['resid_pre', extract_layer][:, pos, :])):
    label_text = np.append(label_text, 'harmless')
    label = np.append(label, 1)

pca = PCA(n_components=3)
fig = make_subplots(rows=6, cols=4,
                    subplot_titles=[f"layer {n}" for n in range(model.cfg.n_layers)])

for row in range(6):
    # print(f'row:{row}')
    for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
        # print(f'layer{layer}')
        if layer <= model.cfg.n_layers:
            activation_layer = torch.cat(
                (harmful_cache['resid_pre', layer][:, pos, :], harmless_cache['resid_pre', layer][:, pos, :]), 0)
            activations_pca = pca.fit_transform(activation_layer.cpu())
            df = {}
            df['label'] = label
            df['pca0'] = activations_pca[:, 0]
            df['pca1'] = activations_pca[:, 1]
            df['label_text'] = label_text

            fig.add_trace(
                go.Scatter(x=df['pca0'],
                           y=df['pca1'],
                           mode='markers',
                           marker_color=df['label'],
                           text=df['label_text']),
                row=row + 1, col=ll + 1,
            )

fig.update_layout(height=1600, width=1000)
fig.show()
fig.write_html(save_path + os.sep  + model_name +'_'+ 'base_with_withouout_urial.html')
fig.write_image(save_path + os.sep  + model_name+'_'+ 'base_with_withouout_urial.png')


#%% 5. Get Direction

harmful_mean_act = harmful_cache['resid_pre', extract_layer][:, pos, :].mean(dim=0)
harmless_mean_act = harmless_cache['resid_pre', extract_layer][:, pos, :].mean(dim=0)

refusal_dir = harmful_mean_act - harmless_mean_act
# refusal_dir = refusal_dir / refusal_dir.norm()

# clean up memory
del harmful_cache, harmless_cache, harmful_logits, harmless_logits
gc.collect(); torch.cuda.empty_cache()

#%% 6. Generation with Hook

def direction_addition_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    strength=1):
    return activation + strength*direction


def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_toks[:, :-max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedy sampling (temperature=0)
            all_toks[:,-max_tokens_generated+i] = next_tokens

    return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

def get_generations(
    model: HookedTransformer,
    instructions: List[str],
    tokenize_instructions_fn: Callable[[List[str]], Int[Tensor, 'batch_size seq_len']],
    fwd_hooks = [],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:

    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations




intervention_dir = refusal_dir
intervention_layers = list(range(model.cfg.n_layers)) # all layers

hook_fn = functools.partial(direction_addition_hook,direction=intervention_dir)
fwd_hooks = [(utils.get_act_name(act_name, extract_layer), hook_fn)  for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

intervention_generations = get_generations(model, harmless_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=fwd_hooks)
baseline_generations = get_generations(model, harmless_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])



for i in range(N_INST_TEST):
    print(f"INSTRUCTION {i}: {repr(harmless_inst_test[i])}")
    print(Fore.GREEN + f"BASELINE COMPLETION:")
    print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)