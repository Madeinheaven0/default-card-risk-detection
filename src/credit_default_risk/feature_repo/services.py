from feast import FeatureService
from views import params_customer


service_model_dispatch = FeatureService(
    name="dispatch_model_v1",
    features=[params_customer],
)