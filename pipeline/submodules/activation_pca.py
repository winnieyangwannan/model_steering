import torch
import os
import json

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_and_cache_activation_addition_output_hook
from pipeline.utils.hook_utils import get_activations_pre_hook



def get_ablation_activations_pre_hook(layer, cache: Float[Tensor, "batch layer d_model"], n_samples, positions: List[int],batch_id,batch_size):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[batch_id:batch_id+batch_size, layer] =  torch.squeeze(activation[:, positions, :],1)
    return hook_fn

def get_addition_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                             direction,
                             batch_size=32, positions=[-1],target_layer=None):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    # if not specified, ablate all layers by default
    if target_layer==None:
        target_layer=np.arange(n_layers)

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        fwd_pre_hooks = [(block_modules[layer],
                          get_and_cache_diff_addition_input_pre_hook(
                                                   direction=direction,
                                                   cache=activations,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer),
                                                ) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations

def get_ablation_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                             direction,
                             batch_size=32, positions=[-1],target_layer=None):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activation_pre = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    # if not specified, ablate all layers by default
    if target_layer==None:
        target_layer=np.arange(n_layers)


    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        fwd_pre_hooks = [(block_modules[layer],
                          get_and_cache_direction_ablation_input_pre_hook(
                                                   direction=direction,
                                                   cache=activation_pre,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer),
                                                ) for layer in range(n_layers)]
        fwd_hooks = [(block_modules[layer],
                          get_and_cache_direction_ablation_output_hook(
                                                   direction=direction,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer),
                                                ) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activation_pre


def get_intervention_activations_and_generation(cfg, model_base, dataset,
                                                tokenize_fn,
                                                mean_diff,
                                                positions=[-1],
                                                target_layer=None,
                                                max_new_tokens=64,
                                                system_type=None,
                                                labels=None):
    torch.cuda.empty_cache()

    model_name = cfg.model_alias
    batch_size = cfg.batch_size
    intervention = cfg.intervention
    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    n_samples = len(dataset)

    # if not specified, ablate all layers by default
    if target_layer == None:
        target_layer = np.arange(n_layers)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_fn(prompts=dataset[i:i+batch_size], system_type=system_type)
        len_inputs = inputs.input_ids.shape[1]

        if intervention == 'direction ablation':
            fwd_pre_hooks = [(block_modules[layer],
                              get_and_cache_direction_ablation_input_pre_hook(
                                                       mean_diff=mean_diff,
                                                       cache=activations,
                                                       layer=layer,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size,
                                                       target_layer=target_layer,
                                                       len_prompt=len_inputs),
                                                    ) for layer in range(n_layers)]
            fwd_hooks = [(block_modules[layer],
                          get_and_cache_direction_ablation_output_hook(
                                                   mean_diff=mean_diff,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer,
                                                   ),
                                                ) for layer in range(n_layers)]
        elif intervention == 'diff addition':
            fwd_pre_hooks = [(block_modules[layer],
                              get_and_cache_diff_addition_input_pre_hook(
                                                       mean_diff=mean_diff,
                                                       cache=activations,
                                                       layer=layer,
                                                       positions=positions,
                                                       batch_id=i,
                                                       batch_size=batch_size,
                                                       target_layer=target_layer,
                                                       len_prompt=len_inputs),
                                                    ) for layer in range(n_layers)]
            fwd_hooks = []

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            generation_toks = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[:, inputs.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                if labels is not None:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                        'label': labels[i + generation_idx]
                    })
                else:
                    completions.append({
                        'prompt': dataset[i + generation_idx],
                        'response': tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })
    return activations, completions


def get_addition_activations_generation(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                             direction,
                             batch_size=32, positions=[-1],target_layer=None,
                            max_new_tokens=64):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    # if not specified, ablate all layers by default
    if target_layer==None:
        target_layer=np.arange(n_layers)

    generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    generation_config.pad_token_id = tokenizer.pad_token_id

    completions = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        len_inputs= inputs.input_ids.shape[1]

        fwd_pre_hooks = [(block_modules[layer],
                          get_and_cache_diff_addition_input_pre_hook(
                                                   direction=direction,
                                                   cache=activations,
                                                   layer=layer,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size,
                                                   target_layer=target_layer,
                                                   len_prompt=len_inputs),
                                                ) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            generation_toks = model.generate(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                generation_config=generation_config,
            )

            generation_toks = generation_toks[:, inputs.input_ids.shape[-1]:]

            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'prompt': instructions[i + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

    return activations, completions


def get_activations(model, block_modules: List[torch.nn.Module],
                    tokenize_fn,
                    dataset,
                    batch_size=32, positions=[-1],
                    system_type=None):

    torch.cuda.empty_cache()

    n_layers = model.config.num_hidden_layers
    n_samples = len(dataset)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_fn(prompts=dataset[i:i+batch_size], system_type=system_type)
        fwd_pre_hooks = [(block_modules[layer],
                          get_activations_pre_hook(layer=layer,
                                                   cache=activations,
                                                   positions=positions,
                                                   batch_id=i,
                                                   batch_size=batch_size)) for layer in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def get_activations_all(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn,
                        block_modules: List[torch.nn.Module],
                        batch_size=32, positions=[-1],
                        intervention="baseline",mean_diff=None,target_layer=None):
    """
    Get layer by layer activations of harmful and harmless instructions

    Returns:
            activations_harmful: Float[Tensor, "n_samples n_layers d_model"]
            activations_harmless: Float[Tensor, "n_samples n_layers d_model"]
    """
    if intervention=="baseline":
        activations_harmful = get_activations(model, harmful_instructions, tokenize_instructions_fn, block_modules,
                                              batch_size=batch_size, positions=positions)
        activations_harmless = get_activations(model, harmless_instructions, tokenize_instructions_fn, block_modules,
                                               batch_size=batch_size, positions=positions)
        completions_harmful = []
        completions_harmless = []

    elif intervention=='ablation':
        activations_harmful, completions_harmful = get_ablation_activations_generation(model, tokenizer,
                                                                                       harmful_instructions, tokenize_instructions_fn,
                                                                                       block_modules,
                                                                                       mean_diff,
                                                                                       batch_size=batch_size, positions=positions,
                                                                                       target_layer=target_layer)
        activations_harmless, completions_harmless = get_ablation_activations_generation(model, tokenizer, harmless_instructions,
                                                                                         tokenize_instructions_fn, block_modules,
                                                                                         mean_diff,
                                                                                         batch_size=batch_size, positions=positions,
                                                                                         target_layer=target_layer)
    elif intervention=='addition':
        # activations_harmful = get_addition_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules,
        #                          mean_diff,
        #                          batch_size=batch_size, positions=positions,target_layer=target_layer)
        # activations_harmless = get_addition_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules,
        #                          mean_diff,
        #                          batch_size=batch_size, positions=positions,target_layer=target_layer)
        activations_harmful, completions_harmful = get_addition_activations_generation(model, tokenizer,
                                                                                       harmful_instructions, tokenize_instructions_fn,
                                                                                       block_modules,
                                                                                       mean_diff,
                                                                                       batch_size=batch_size, positions=positions,
                                                                                       target_layer=target_layer)
        activations_harmless, completions_harmless = get_addition_activations_generation(model, tokenizer,
                                                                                         harmless_instructions,
                                                                                         tokenize_instructions_fn, block_modules,
                                                                                         mean_diff,
                                                                                         batch_size=batch_size, positions=positions,
                                                                                         target_layer=target_layer)

    return activations_harmful,activations_harmless,completions_harmful, completions_harmless



def generate_activations_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    batch_size = cfg.batch_size

    activations_harmful,activations_harmless,_,_ = get_activations_all(model_base.model,
                                                                       model_base.tokenizer,
                                                                       harmful_instructions,
                                                                       harmless_instructions,
                                                                       model_base.tokenize_instructions_fn,
                                                                       model_base.model_block_modules,
                                                                       positions=[-1],
                                                                       batch_size=batch_size,
                                                                       intervention="baseline")

    # plot pca
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_harmful,activations_harmless,n_layers)
    model_name =cfg.model_alias
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'activation_pca.html')

    return activations_harmful,activations_harmless

def refusal_intervention_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions,mean_diff,
                                 ):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    intervention = cfg.intervention
    source_layer = cfg.source_layer
    target_layer = cfg.target_layer
    batch_size = cfg.batch_size


    #
    # activations_harmful,activations_harmless = get_activations_all(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions,
    #                          model_base.tokenize_instructions_fn, model_base.model_block_modules,
    #                          positions=[-1],
    #                          batch_size=batch_size,
    #                          intervention=intervention,mean_diff=mean_diff[source_layer],target_layer=target_layer
    #                           )
    activations_harmful,activations_harmless, completions_harmful, completions_harmless = get_activations_all(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions,
                             model_base.tokenize_instructions_fn, model_base.model_block_modules,
                             positions=[-1],
                             batch_size=batch_size,
                             intervention=intervention,mean_diff=mean_diff[source_layer],target_layer=target_layer
                              )

    # save completion
    harmless_completion_path = f'{cfg.artifact_path()}' + os.sep + f'completions_harmless_{intervention}_completions.json'
    harmful_completion_path = f'{cfg.artifact_path()}' + os.sep + f'completions_harmful_{intervention}_completions.json'

    with open(harmless_completion_path, "w") as f:
        json.dump(completions_harmless, f, indent=4)
    with open(harmful_completion_path, "w") as f:
        json.dump(completions_harmful, f, indent=4)

    # PCA
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_harmful,activations_harmless,n_layers)
    model_name =cfg.model_alias
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_'+intervention+'_pca_layer_'+str(source_layer)+'_'+ str(target_layer)+'.html')

    return activations_harmful, activations_harmless


def plot_contrastive_activation_pca(activations_harmful, activations_harmless, n_layers, contrastive_label,
                                    labels=None):

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_harmful,
                                                                              activations_harmless), dim=0)

    n_data = activations_harmless.shape[0]

    if labels is not None:
        labels_all = labels + labels
        labels_t = []
        for ll in labels:
            if ll == 0:
                labels_t.append('false')
            elif ll == 1:
                labels_t.append('true')
    else:
        labels_all = np.zeros((n_data*2),1)

    label_text = []
    for ii in range(n_data):
        label_text = np.append(label_text, f'honest_{labels_t[ii]}_{ii}')
    for ii in range(n_data):
        label_text = np.append(label_text, f'lying_{labels_t[ii]}_{ii}')


    cols = 4
    rows = int(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])

    pca = PCA(n_components=3)

    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer <= n_layers:
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = labels_all
                df['pca0'] = activations_pca[:, 0]
                df['pca1'] = activations_pca[:, 1]
                df['label_text'] = label_text

                fig.add_trace(
                    go.Scatter(x=df['pca0'][:n_data],
                               y=df['pca1'][:n_data],
                               mode="markers",
                               name="honest",
                               showlegend=False,
                               marker=dict(
                                   symbol="star",
                                   size=8,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color=df['label'][:n_data]
                               ),
                           text=df['label_text'][:n_data]),
                           row=row+1, col=ll+1,
                            )
                fig.add_trace(
                    go.Scatter(x=df['pca0'][n_data:],
                               y=df['pca1'][n_data:],
                               mode="markers",
                               name="lying",
                               showlegend=False,
                               marker=dict(
                                   symbol="circle",
                                   size=5,
                                   line=dict(width=1, color="DarkSlateGrey"),
                                   color=df['label'][n_data:],
                               ),
                           text=df['label_text'][n_data:]),
                           row=row+1, col=ll+1,
                           )
    # legend
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_data:],
                   ),
                   name=f'honest_false',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="star",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_data:],
                   ),
                   name=f'honest_true',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_data:],
                   ),
                   name=f'lying_false',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   marker=dict(
                       symbol="circle",
                       size=5,
                       line=dict(width=1, color="DarkSlateGrey"),
                       color=df['label'][n_data:],
                   ),
                   name=f'lying_true',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.update_layout(height=1600, width=1000)
    fig.show()
    fig.write_html('honest_lying_pca.html')

    return fig


def plot_contrastive_activation_intervention_pca(activations_harmful,
                                                 activations_harmless,
                                                 ablation_activations_harmful,
                                                 ablation_activations_harmless,
                                                 n_layers,
                                                 contrastive_label,
                                                 labels=None,
                                                 ):

    activations_all: Float[Tensor, "n_samples n_layers d_model"] = torch.cat((activations_harmful,
                                                                              activations_harmless,
                                                                              ablation_activations_harmful,
                                                                              ablation_activations_harmless),
                                                                             dim=0)

    pca = PCA(n_components=3)

    label_text = []
    label_plot = []

    for ii in range(activations_harmful.shape[0]):
        label_text = np.append(label_text, f'{contrastive_label[0]}:{ii}')
        label_plot.append(0)
    for ii in range(activations_harmless.shape[0]):
        label_text = np.append(label_text, f'{contrastive_label[1]}:{ii}')
        label_plot.append(1)
    for ii in range(activations_harmful.shape[0]):
        label_text = np.append(label_text, f'{contrastive_label[2]}:{ii}')
        label_plot.append(2)
    for ii in range(activations_harmful.shape[0]):
        label_text = np.append(label_text, f'{contrastive_label[3]}:{ii}')
        label_plot.append(3)
    cols = 4
    rows = int(n_layers/cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"layer {n}" for n in range(n_layers)])
    for row in range(rows):
        # print(f'row:{row}')
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer <= n_layers:
                activations_pca = pca.fit_transform(activations_all[:, layer, :].cpu())
                df = {}
                df['label'] = label_plot
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
    # legend
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_label[0]}',
                   marker_color='blue',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_label[1]}',
                   marker_color='purple',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_label[2]}',
                   marker_color='orange',
                 ),
        row=row + 1, col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(x=[None],
                   y=[None],
                   mode='markers',
                   name=f'{contrastive_label[3]}',
                   marker_color='yellow',
                 ),
        row=row + 1, col=ll + 1,
    )

    fig.update_layout(
        showlegend=True
    )
    fig.update_layout(height=1600, width=1000)
    fig.show()
    return fig


def extraction_intervention_and_plot_pca(cfg,model_base: ModelBase, harmful_instructions, harmless_instructions):
    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # 1. extract activations
    activations_harmful,activations_harmless = generate_activations_and_plot_pca(cfg,model_base, harmful_instructions, harmless_instructions)

    # 2. get steering vector = get mean difference of the source layer
    mean_activation_harmful = activations_harmful.mean(dim=0)
    mean_activation_harmless = activations_harmless.mean(dim=0)
    mean_diff = mean_activation_harmful-mean_activation_harmless


    # 3. refusal_intervention_and_plot_pca
    ablation_activations_harmful,ablation_activations_harmless = refusal_intervention_and_plot_pca(cfg, model_base,
                                                                                                   harmful_instructions,
                                                                                                   harmless_instructions,
                                                                                                   mean_diff)

    # 4. pca with and without intervention, plot and save pca
    intervention = cfg.intervention
    source_layer = cfg.source_layer
    target_layer = cfg.target_layer
    model_name =cfg.model_alias
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_intervention_pca(activations_harmful, activations_harmless,
                                                       ablation_activations_harmful, ablation_activations_harmless,
                                                       n_layers)
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_generation_activation_'
                   +intervention+'_pca_layer_'
                   +str(source_layer)+'_'+ str(target_layer)+'.html')