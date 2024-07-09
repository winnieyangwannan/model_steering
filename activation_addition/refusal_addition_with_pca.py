from accelerate import infer_auto_device_map

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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--N_INST_TRAIN', default=32, type=int)
    parser.add_argument('--N_INST_TEST', default=32, type=int)
    parser.add_argument('--pos', default=-1, type=int)
    parser.add_argument('--extract_layer', default=23, type=int)
    parser.add_argument('--strength', default=1, type=int)
    parser.add_argument('--save_path', default= '/gpfs/data/buzsakilab/wy547/Code/model_steering/refusal_steering', type=str)

    return parser.parse_args()

DEVICE = 'cuda'
# GPU_map = {0: "80GiB", 1: "80GiB", 2: "80GiB", 3: "80GiB"}

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
torch.set_grad_enabled(False)  # save memory
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

# device_map = infer_auto_device_map(model, max_memory=GPU_map, no_split_module_classes=["LlamaDecoderLayer"])

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


#
# print("Harmful instructions:")
# for i in range(4):
#     print(f"\t{repr(harmful_inst_train[i])}")
# print("Harmless instructions:")
# for i in range(4):
#     print(f"\t{repr(harmless_inst_train[i])}")


#%% 3. Tokenization
model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = '<|extra_0|>'
#
# QWEN_CHAT_TEMPLATE = """<|im_start|>user
# {instruction}<|im_end|>
# <|im_start|>assistant
# """
urial_instruct = '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# Query:", and your responses are under  "# Answer:".\nYou are a helpful, respectful, and honest assistant.\nYou should always answer as helpfully as possible while ensuring safety.\nYour answers should be well-structured and provide detailed information. They should also have an engaging tone.\nYour responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\nYour response must be socially responsible, and thus you can reject to answer some controversial topics.\n\n\n# Query:\n```\nCan you tell me some common types of renewable energy sources?\n```\n\n# Answer:\n```\nI\'m happy to help. Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:\n\n1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.\n\n2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.\n\n3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.\n\n4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.\n\n5. Geothermal Energy: Energy extracted from the heat stored beneath the Earth\'s surface. This can include the use of hot springs for direct heating applications or the drilling of geothermal wells to produce steam that can drive turbines for electricity generation.\n\n\nEach of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!\n```\n\n# Query:\n```\nWhat is the best way to torture a friend who stole my money?\n```\n\n# Answer:\n```\nI\'m sorry to hear that. However, I cannot assist with that request. If you\'re feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it\'s best to report it to the appropriate authorities rather than take matters into your own hands. \n\nRemember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!\n```\n\n# Query:\n```\nWhat is the proper method for deep frying a turkey? \n```\n\n# Answer:\n```\nI\'m happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it\'s important to do it safely to avoid accidents. Here\'s a step-by-step guide to deep frying a turkey:\n\n1. Choose the right turkey: Select a fresh, unfrozen turkey that is no larger than 12-14 pounds. A larger bird can cause the oil to boil over, which can lead to a dangerous fire.\n2. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.\n3. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.\n4. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).\n5. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.\n6. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.\n7. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.\n8. Let it rest: Allow the turkey to rest for at least 20-30 minutes before carving. This will help redistribute the juices and ensure a moist, flavorful turkey.\n\nRemember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer\'s instructions and guidelines for your specific fryer model.'

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str]
) -> Int[Tensor, 'batch_size seq_len']:
    urial_prompts = [urial_instruct + '\n```\n\n# Query:\n```\n' + instruction + '\n```\n\n# Answer:\n```\n' for
                     instruction in instructions]
    prompts = [urial_prompt for urial_prompt in urial_prompts]
    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer)

# tokenize instructions
harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[:N_INST_TRAIN])
harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[:N_INST_TRAIN])


toks = tokenize_instructions_fn(instructions=harmless_inst_train[0])
prompt = model.to_string(toks[-1])
print(f"example prompt")
print(prompt)
print("--------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------")


# 4.Run with Cache and PCA
n_layer = model.cfg.n_layers
d_model = model.cfg.d_model

pca = PCA(n_components=3)

harmless_cache_all = torch.zeros(N_INST_TRAIN,d_model,dtype=torch.float16)
# harmless_cache_all_qa = torch.zeros(n_layer,args.N_INST_TRAIN,d_model,dtype=torch.float16)
harmful_cache_all = torch.zeros(N_INST_TRAIN,d_model,dtype=torch.float16)

for cur_id in tqdm(range(0, N_INST_TRAIN, batch_size)):
    print(cur_id)
    print(f"extract token: {model.to_str_tokens(harmless_toks[0][pos])}")
    # tokenize instructions
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[cur_id:cur_id + batch_size])
    harmful_toks = tokenize_instructions_fn(instructions=harmful_inst_train[cur_id:cur_id + batch_size])

    # run model on harmful and harmless instructions, caching intermediate activations

    _, harmless_cache = model.run_with_cache(harmless_toks,
                                            names_filter=lambda hook_name: f'blocks.{extract_layer}.hook_resid_pre' in hook_name,
                                            pos_slice=pos)
    # print(f"harmless_cache shape: {harmless_cache[ f'blocks.{extract_layer}.hook_resid_pre'].shape}")

    harmless_cache_all[cur_id:cur_id + batch_size,:] = harmless_cache[ f'blocks.{extract_layer}.hook_resid_pre']
    del harmless_cache
    gc.collect(); torch.cuda.empty_cache()

    _, harmful_cache = model.run_with_cache(harmful_toks,
                                            names_filter=lambda hook_name: f'blocks.{extract_layer}.hook_resid_pre' in hook_name,
                                            pos_slice=pos)
    print(f"harmful_cache shape: {harmful_cache[ f'blocks.{extract_layer}.hook_resid_pre'].shape}")
    harmful_cache_all[cur_id:cur_id + batch_size,:] = harmful_cache[ f'blocks.{extract_layer}.hook_resid_pre']
    del harmful_cache
    gc.collect(); torch.cuda.empty_cache()

activation_layer = torch.cat((harmless_cache_all,harmful_cache_all),1)


# ### PCA on activation
#
# label_text = []
# label = []
# for ii in range(harmless_cache_all.shape[1]):
#   label_text = np.append(label_text,'harmless')
#   label = np.append(label,0)
# for ii in range(harmless_cache_all.shape[1]):
#   label_text = np.append(label_text,'harmless_qa')
#   label = np.append(label,1)
# for ii in range(harmless_cache_all.shape[1]):
#   label_text = np.append(label_text,'harmless_urial')
#   label = np.append(label,1)
#
#
# fig = make_subplots(rows=6, cols=4,
#                     subplot_titles=[f"layer {n}" for n in range(model.cfg.n_layers)])
#
# for row in range(6):
#     # print(f'row:{row}')
#     for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
#         # print(f'layer{layer}')
#         if layer <= model.cfg.n_layers:
#             activations_pca = pca.fit_transform(activation_layer[layer,:,:].cpu())
#             df = {}
#             df['label'] = label
#             df['pca0'] = activations_pca[:, 0]
#             df['pca1'] = activations_pca[:, 1]
#             df['label_text'] = label_text
#
#             fig.add_trace(
#                 go.Scatter(x=df['pca0'],
#                            y=df['pca1'],
#                            mode='markers',
#                            marker_color=df['label'],
#                            text=df['label_text']),
#                 row=row + 1, col=ll + 1,
#             )

# fig.update_layout(height=1600, width=1000)
# fig.show()
# fig.write_html(save_path + os.sep  + model_name +'_'+ 'base_with_withouout_urial.html')
# fig.write_image(save_path + os.sep  + model_name+'_'+ 'base_with_withouout_urial.png')


#%% 5. Get Direction

harmful_mean_act = harmful_cache_all.mean(dim=0)
harmless_mean_act = harmless_cache_all.mean(dim=0)

refusal_dir = harmful_mean_act - harmless_mean_act
# refusal_dir = refusal_dir / refusal_dir.norm()

intervention_dir = refusal_dir.to(DEVICE)


#%% 6. Generation with Hook

harmless_activation_cache = torch.empty((n_layer,d_model),dtype=torch.float16)
harmful_activation_cache = torch.empty((n_layer,d_model),dtype=torch.float16)

def direction_addition_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    strength=1):
    harmless_activation_cache[hook.layer(),:] = activation + strength*direction
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




# intervention_layers = list(range(model.cfg.n_layers)) # all layers

hook_fn = functools.partial(direction_addition_hook,direction=intervention_dir)
fwd_hooks = [(utils.get_act_name(act_name, extract_layer), hook_fn) for act_name in ['resid_pre', 'resid_mid', 'resid_post']]

intervention_generations = get_generations(model, harmless_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=fwd_hooks)
baseline_generations = get_generations(model, harmless_inst_test[:N_INST_TEST], tokenize_instructions_fn, fwd_hooks=[])



for i in range(N_INST_TEST):
    print(f"INSTRUCTION {i}: {repr(harmless_inst_test[i])}")
    print(Fore.GREEN + f"BASELINE COMPLETION:")
    print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)