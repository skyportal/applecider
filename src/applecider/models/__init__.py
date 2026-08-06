from .astrominn import AstroMiNN
from .HyraxBaselineCLS import HyraxBaselineCLS, MPTModel
from .spectranet import SpectraNet

# make all the `hyrax_model` decorated models available for import from applecider.models
__all__ = ["AstroMiNN", "HyraxBaselineCLS", "MPTModel", "SpectraNet"]