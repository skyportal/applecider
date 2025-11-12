from hyrax import Hyrax

toml_path = "/Users/dbranton/lincc/incubators/applecider/notebooks/testing/baselinecls_testing_runtime_config.toml"
h = Hyrax(config_file=toml_path)

dataset = h.prepare()
h.train()
h.infer()
