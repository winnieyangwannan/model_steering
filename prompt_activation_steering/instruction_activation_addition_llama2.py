import torch
import functools
import requests
import pandas as pd
import io
import textwrap
import gc
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import argparse
import random

from datasets import load_dataset
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
import numpy as np

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens.hook_points import HookPoint
from colorama import Fore
import numpy as np
from sklearn.decomposition import PCA

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
### Load model Here


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-2-7b-hf", type=str)
    parser.add_argument('--model_path', default="/gpfs/data/buzsakilab/wy547/DATA/Llama/Llama_2/Llama-2-7b-hf", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--N_INST_TRAIN', default=32, type=int)
    parser.add_argument('--N_INST_TEST', default=32, type=int)
    parser.add_argument('--pos', default=-1, type=int)
    parser.add_argument('--layer', default=23, type=int)
    parser.add_argument('--strength', default=10, type=int)
    parser.add_argument('--save_path', default= '/gpfs/data/buzsakilab/wy547/Code/model_steering/prompt_activation_steering', type=str)

    return parser.parse_args()

DEVICE = 'cuda'
args = parse_args()
model_name = args.model.split('/')[-1]
print(f"model: {model_name}")
print(f"batch_size: {args.batch_size}")
print(f"N_INST_TRAIN: {args.N_INST_TRAIN}")
print(f"N_INST_TEST: {args.N_INST_TEST}")
print(f"pos: {args.pos}")
print(f"layer: {args.layer}")
print(f"strength: {args.strength}")
print(f"save_path: {args.save_path}")



### 1. Load Model
inference_dtype = torch.float16
DEVICE = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                          force_download=True,
                                          trust_remote_code=True,
                                          device_map="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id  # eos_token_id=50256

if "gpt" in model_name:
    tokenizer.padding_side = 'left'


print("Loading with quantization...")
hf_model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                device_map="auto",
                                                torch_dtype=inference_dtype,
                                                force_download=True,
                                                trust_remote_code=True)
                                                # quantization_config=quantization_config) #torch_dtype=inference_dtype,
#quantization_config = quantization_config,
#torch_dtype = inference_dtype
model = HookedTransformer.from_pretrained(args.model,
                                             hf_model=hf_model,
                                             dtype=inference_dtype,
                                             fold_ln=False,
                                             fold_value_biases=False,
                                             center_writing_weights=False,
                                             center_unembed=False,
                                             tokenizer=tokenizer,
                                             device=DEVICE,
                                             )

del hf_model
go.collect(); torch.cuda.empty_cache()

#%%
# 2. Load Dataset


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



random.seed(10)
harmless_inst_train, harmless_inst_test = get_harmless_instructions()

#%%
# 3. Tokenization
model.tokenizer.padding_side = 'left'
model.tokenizer.pad_token = '<|extra_0|>'
urial_instruct = '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# Query:", and your responses are under  "# Answer:".\nYou are a helpful, respectful, and honest assistant.\nYou should always answer as helpfully as possible while ensuring safety.\nYour answers should be well-structured and provide detailed information. They should also have an engaging tone.\nYour responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\nYour response must be socially responsible, and thus you can reject to answer some controversial topics.\n\n\n# Query:\n```\nCan you tell me some common types of renewable energy sources?\n```\n\n# Answer:\n```\nI\'m happy to help. Certainly! Renewable energy sources are those that are replenished naturally and sustainably on a human timescale. Here are some common types of renewable energy sources:\n\n1. Solar Energy: Energy harnessed from the sun. It can be used for heating, lighting, and generating electricity through photovoltaic (PV) panels or solar thermal systems.\n\n2. Wind Energy: Energy generated from the natural movement of air. Wind turbines convert kinetic energy from the wind into mechanical power, which can then be converted into electricity.\n\n3. Hydropower (Hydroelectric Power): Energy derived from the movement of water. This can be the flow of rivers, tidal movements, or wave power. Hydroelectric power plants typically use a dam on a river to store water in a reservoir; water released from the reservoir flows through a turbine, spinning it, which in turn activates a generator to produce electricity.\n\n4. Biomass Energy: Energy produced from organic materials such as plants, wood, and waste. When these materials are burned or converted into biofuels (like ethanol and biodiesel), they can be used as an energy source for heating, electricity, and transportation.\n\n5. Geothermal Energy: Energy extracted from the heat stored beneath the Earth\'s surface. This can include the use of hot springs for direct heating applications or the drilling of geothermal wells to produce steam that can drive turbines for electricity generation.\n\n\nEach of these renewable energy sources has its own advantages and challenges in terms of technology, cost, and the potential impact on the environment. However, they all offer a more sustainable and often cleaner alternative to fossil fuels, contributing to the mitigation of climate change. Please let me know if you have any other questions!\n```\n\n# Query:\n```\nWhat is the best way to torture a friend who stole my money?\n```\n\n# Answer:\n```\nI\'m sorry to hear that. However, I cannot assist with that request. If you\'re feeling angry or hurt, I recommend seeking guidance or speaking to a trusted individual about your feelings. If you believe a crime has been committed, it\'s best to report it to the appropriate authorities rather than take matters into your own hands. \n\nRemember, violence or harm is not a solution and can lead to serious consequences for both parties involved. I hope it helps!\n```\n\n# Query:\n```\nWhat is the proper method for deep frying a turkey? \n```\n\n# Answer:\n```\nI\'m happy to help you with your question about deep frying a turkey! Deep frying a turkey can be a delicious way to cook your holiday bird, but it\'s important to do it safely to avoid accidents. Here\'s a step-by-step guide to deep frying a turkey:\n\n1. Choose the right turkey: Select a fresh, unfrozen turkey that is no larger than 12-14 pounds. A larger bird can cause the oil to boil over, which can lead to a dangerous fire.\n2. Prepare the turkey: Thoroughly thaw and dry the turkey before frying. Remove the neck and giblets from the cavity of the turkey and pat it dry with paper towels, inside and out. This will help prevent oil splatter and ensure a crispy skin.\n3. Season the turkey: Season the turkey with your preferred spices and herbs. You can use a dry rub or inject the seasoning directly into the meat.\n4. Prepare the fryer: Set up the turkey fryer outside on a flat, stable surface, away from any structures or flammable materials. Fill the fryer with peanut or canola oil to the 1. recommended level, typically indicated on the fryer. Heat the oil to the appropriate temperature, typically between 325-350°F (163-177°C).\n5. Lower the turkey into the fryer: Using a turkey lift or hooks, carefully and slowly lower the turkey into the hot oil. Make sure the turkey is fully submerged in the oil.\n6. Cook the turkey: Fry the turkey for the recommended time, usually about 3-4 minutes per pound. Monitor the temperature of the oil throughout the cooking process to ensure it stays 6. within the recommended range.\n7. Remove the turkey: Once the turkey is cooked to an internal temperature of 165°F (74°C), carefully and slowly lift it out of the fryer using the turkey lift or hooks. Place it on a wire rack or tray to drain any excess oil.\n8. Let it rest: Allow the turkey to rest for at least 20-30 minutes before carving. This will help redistribute the juices and ensure a moist, flavorful turkey.\n\nRemember to always prioritize safety when deep frying a turkey. Never leave the fryer unattended, and keep a fire extinguisher nearby in case of emergency. Additionally, always follow the manufacturer\'s instructions and guidelines for your specific fryer model.'
qa_instruct = '# Instruction\n\nBelow is a list of conversations between a human and an AI assistant (you). \nUsers place their queries under "# Query:", and your responses are under  "# Answer:".'

print(urial_instruct)
print('-------------------------------------------------------')
print('-------------------------------------------------------')
print('-------------------------------------------------------')
print(qa_instruct)


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    instruct= 'base'
) -> Int[Tensor, 'batch_size seq_len']:
    if instruct == 'qa':
        prompts = [qa_instruct + '\n```\n\n# Query:\n```\n' + instruction + '\n```\n\n# Answer:\n```\n' for instruction in instructions]
    elif instruct == 'urial':
        prompts = [urial_instruct + '\n```\n\n# Query:\n```\n' + instruction + '\n```\n\n# Answer:\n```\n' for instruction in instructions]
    elif instruct == 'base':
        prompts = [instruction for instruction in instructions]

    return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

tokenize_instructions_fn = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer, instruct='base')
# tokenize_instructions_fn_qa = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer, instruct='qa')
tokenize_instructions_fn_urial = functools.partial(tokenize_instructions_qwen_chat, tokenizer=model.tokenizer,instruct='urial')


# 4.Run with Cache and PCA
n_layer = model.cfg.n_layers
d_model = model.cfg.d_model

pca = PCA(n_components=3)

harmless_cache_all = torch.zeros(n_layer,args.N_INST_TRAIN,d_model,dtype=torch.float16)
# harmless_cache_all_qa = torch.zeros(n_layer,args.N_INST_TRAIN,d_model,dtype=torch.float16)
harmless_cache_all_urial = torch.zeros(n_layer,args.N_INST_TRAIN,d_model,dtype=torch.float16)

for cur_id in tqdm(range(0, args.N_INST_TRAIN, args.batch_size)):
    # tokenize instructions
    harmless_toks = tokenize_instructions_fn(instructions=harmless_inst_train[cur_id:cur_id + args.batch_size])
    # harmless_toks_qa = tokenize_instructions_fn_qa(instructions=harmless_inst_train[cur_id:cur_id + args.batch_size])
    harmless_toks_urial = tokenize_instructions_fn_urial(instructions=harmless_inst_train[cur_id:cur_id + args.batch_size])

    # run model on harmful and harmless instructions, caching intermediate activations
    harmless_logits, harmless_cache = model.run_with_cache(harmless_toks,
                                                           names_filter=lambda hook_name: 'resid' in hook_name)
    # harmless_logits_qa, harmless_cache_qa = model.run_with_cache(harmless_toks_qa,
    #                                                        names_filter=lambda hook_name: 'resid' in hook_name)
    harmless_logits_urial, harmless_cache_urial = model.run_with_cache(harmless_toks_urial,
                                                           names_filter=lambda hook_name: 'resid' in hook_name)

    for layer in range(model.cfg.n_layers):
        harmless_cache_all[layer,cur_id:cur_id + args.batch_size,:] = harmless_cache['resid_pre', layer][:, args.pos, :]
        # harmless_cache_all_qa[layer,cur_id:cur_id + args.batch_size,:] = harmless_cache_qa['resid_pre', layer][:, args.pos, :]
        harmless_cache_all_urial[layer,cur_id:cur_id + args.batch_size,:] = harmless_cache_urial['resid_pre', layer][:, args.pos, :]

# activation_layer = torch.cat((harmless_cache_all,harmless_cache_all_qa,harmless_cache_all_urial),1)
activation_layer = torch.cat((harmless_cache_all,harmless_cache_all_urial),1)


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
#
# fig.update_layout(height=1600, width=1000)
# fig.show()
# fig.write_html(args.save_path + os.sep + model_name +'_'+ 'base_with_withouout_urial.html')
# fig.write_image(args.save_path + os.sep +model_name +'_'+ 'base_with_withouout_urial.png')

# 5.Finding the "instruction direction"


harmless_mean_act = harmless_cache_all[args.layer,:,:].mean(dim=0)
harmless_mean_act_urial = harmless_cache_all_urial[args.layer,:,:].mean(dim=0)
instruct_dir = harmless_mean_act_urial - harmless_mean_act

# clean up memory
# del harmless_cache,harmless_cache_qa, harmless_cache_urial, harmless_logits_urial, harmless_logits_qa,harmless_logits
del harmless_cache, harmless_cache_urial, harmless_logits_urial,harmless_logits

gc.collect(); torch.cuda.empty_cache()

def direction_addition_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
    strength=1):
    return activation + strength*direction

intervention_dir = instruct_dir.to(DEVICE)
intervention_layers = list(range(model.cfg.n_layers)) # all layers

hook_fn = functools.partial(direction_addition_hook,direction=intervention_dir,strength=args.strength)
# all layers
# fwd_hooks = [(utils.get_act_name(act_name, l), hook_fn) for l in intervention_layers for act_name in ['resid_pre', 'resid_mid', 'resid_post']]
# only one layer
fwd_hooks = [(utils.get_act_name(act_name, args.layer), hook_fn) for act_name in ['resid_pre', 'resid_mid', 'resid_post']]


# 6.Generation with hook

def _generate_with_hooks(
    model: HookedTransformer,
    toks: Int[Tensor, 'batch_size seq_len'],
    max_tokens_generated: int = 64,
    fwd_hooks = [],
) -> List[str]:

    all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=toks.device)
    all_toks[:, :toks.shape[1]] = toks

    for i in range(max_tokens_generated):
        # ALL TOKEN POSITIONS
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
    batch_size: int = 2,
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

intervention_generations = get_generations(model,
                                           harmless_inst_train[:args.N_INST_TEST],
                                           tokenize_instructions_fn,
                                           fwd_hooks=fwd_hooks,
                                           batch_size=args.batch_size)

# qa_generations = get_generations(model,
#                                  harmless_inst_train[:args.N_INST_TEST],
#                                  tokenize_instructions_fn_qa,
#                                  fwd_hooks=[],
#                                  batch_size=args.batch_size)

baseline_generations = get_generations(model,
                                       harmless_inst_train[:args.N_INST_TEST],
                                       tokenize_instructions_fn,
                                       fwd_hooks=[],
                                       batch_size=args.batch_size)

#%%
# 7. Print Results

for i in range(args.N_INST_TEST):
    print(f"INSTRUCTION {i}: {repr(harmless_inst_train[i])}")
    print(Fore.GREEN + f"BASELINE COMPLETION:")
    print(textwrap.fill(repr(baseline_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    # print(Fore.RED + f"QA COMPLETION:")
    # print(textwrap.fill(repr(qa_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RED + f"INTERVENTION COMPLETION:")
    print(textwrap.fill(repr(intervention_generations[i]), width=100, initial_indent='\t', subsequent_indent='\t'))
    print(Fore.RESET)