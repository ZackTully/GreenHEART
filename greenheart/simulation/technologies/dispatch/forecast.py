import numpy as np
import matplotlib.pyplot as plt

class Forecast:

    generation_profile: np.ndarray
    # resource_profile: np.ndarray # Ignore for now

    forecast_horizon: int # number of timesteps to 
    forecast_method: str

    def __init__(self):
        pass

    def get_forecast(self, step_index):
        forecast = 10
        return forecast