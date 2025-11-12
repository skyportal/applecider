from hyrax import Hyrax

toml_path = "/Users/dbranton/lincc/incubators/applecider/notebooks/testing/baselinecls_testing_runtime_config.toml"
h = Hyrax(config_file=toml_path)
#h.set_config("model_inputs.data.dataset_class", "AppleCider.models.hyrax_models.photo_dataset.PhotoEventsDataset")

dataset = h.prepare()
h.train()
h.infer()