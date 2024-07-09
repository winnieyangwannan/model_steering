import torch
import random
import json
import os
import argparse


from datasets import load_dataset
from torch.utils.data import DataLoader
from pipeline.submodules.evaluate_truthful import plot_lying_honest_accuracy, get_statement_accuracy_cache_activation
from pipeline.honesty_pipeline.honesty_config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
# from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        train and test dataset
    """

    random.seed(42)

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    data = random.sample(dataset, cfg.n_extract)

    return data

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def get_lying_honest_accuracy_and_plot(cfg, model_base, dataset):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    model_name = cfg.model_alias

    # lying prompt template
    accuracy_lying, probability_lying, unexpected_lying, activations_lying = get_statement_accuracy_cache_activation(
                                                                                 model_base,
                                                                                 dataset,
                                                                                 cfg, system_type="lying",)

    # honest prompt template
    accuracy_honest, probability_honest, unexpected_honest, activations_honest = get_statement_accuracy_cache_activation(
                                                                                model_base,
                                                                                dataset,
                                                                                cfg, system_type="honest",)

    print(f"accuracy_lying: {sum(accuracy_lying)/len(accuracy_lying)}")
    print(f"accuracy_honest: {sum(accuracy_honest)/len(accuracy_lying)}")
    print(f"unexpected_lying: {sum(unexpected_lying)/len(accuracy_lying)}")
    print(f"unexpected_honest: {sum(unexpected_honest)/len(accuracy_lying)}")

    # plot activation pca
    contrastive_label = ["honest", "lying"]
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_honest, activations_lying, n_layers, contrastive_label)
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'honest_lying_pca.html')

    # plot
    plot_lying_honest_accuracy()


def run_pipeline(model_path):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset = load_and_sample_datasets(cfg)

    # 3. Get Accuracy
    get_lying_honest_accuracy_and_plot(cfg, model_base, dataset)
    

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)