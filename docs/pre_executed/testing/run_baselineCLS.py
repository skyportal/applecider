from hyrax import Hyrax
from hyrax.config_utils import find_most_recent_results_dir
import os

toml_path = "./baselinecls_testing_runtime_config.toml"
h = Hyrax(config_file=toml_path)

# For training


# For inference
#h.set_config("model_inputs.data.dataset_class", "AppleCider.models.hyrax_models.photo_dataset.PhotoEventsDataset")

dataset = h.prepare()
'''
epochs = 21
for i, ep in enumerate(range(epochs)):
    # Training
    h.set_config("model.HyraxBaselineCLS.use_probabilities", False)
    if i > 0:
        train_dir = find_most_recent_results_dir(h.config, "train")
        epoch_checkpoint = train_dir / f"checkpoint_epoch_{i}.pt"
        h.set_config("train.resume", str(epoch_checkpoint))
    h.set_config("train.epochs", ep+1)
    h.train()

    # Inference every 3
    if ep % 3 == 0:
        h.set_config("model.HyraxBaselineCLS.use_probabilities", True)
        h.infer()

# Training
'''

h.set_config("model.HyraxBaselineCLS.use_probabilities", False)
h.train()

# Inference
h.set_config("model.HyraxBaselineCLS.use_probabilities", True)
h.infer()
