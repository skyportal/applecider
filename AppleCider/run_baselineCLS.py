from hyrax import Hyrax

toml_path = "/Users/dbranton/lincc/incubators/baselineCLS_runtime_config.toml"
h = Hyrax(config_file=toml_path)
h.config["model_inputs"] = {
    "data": {
        "dataset_class": "AppleCider.models.hyrax_models.photo_dataset.PhotoEventsDataset",
        "data_location": "/Users/dbranton/lincc/incubators/photo_events/train/",
        "fields": ["photometry", "label"],
    }
}

dataset = h.prepare()
h.train()