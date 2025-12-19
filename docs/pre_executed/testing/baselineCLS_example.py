from hyrax import Hyrax
from hyrax.config_utils import find_most_recent_results_dir
import os
import torch
from hyrax.pytorch_ignite import (
            setup_model,
        )

# Path definitions
toml_path = "./baselinecls_example_runtime_config.toml"


h = Hyrax(config_file=toml_path)
dataset = h.prepare()

# Perform Pre-Training
perform_pretrain = False # Can also be grabbed from config
if perform_pretrain:
    pretrain_weights_path = h.config["model"]["HyraxBaselineCLS"]["pretrained_weights_path"]
    # Prepare empty state_dict for model
    model = setup_model(h.config, dataset["train"])
    initial_state_dict = model.state_dict()

    # Perform pre-training
    pretrainer = Hyrax(config_file=toml_path)
    pretrainer.set_config("model.name", "applecider.models.HyraxBaselineCLS.MPTModel")
    pretrain = pretrainer.train() # Train the pre-trainer
    # Save output weights to file
    weights = pretrain.state_dict()

    # Override matching weight fields in the initial model state dict
    for k, v in weights.items():
        if k.startswith("head."):  # skip classifier head
            continue
        if k in initial_state_dict and initial_state_dict[k].shape == v.shape:
            initial_state_dict[k] = v
    torch.save(initial_state_dict, pretrain_weights_path)


# Training
h.set_config("model.HyraxBaselineCLS.use_probabilities", False)
h.set_config("data_set.PhotoEventsDataset.use_oversampling", True)
h.train()

# Inference
h.set_config("model.HyraxBaselineCLS.use_probabilities", True)
h.set_config("data_set.PhotoEventsDataset.use_oversampling", False) # Disable oversampling for inference
h.infer()