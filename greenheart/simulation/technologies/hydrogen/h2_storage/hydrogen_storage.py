import numpy as np


class HydrogenStorage:
    def __init__(self):
        self.eta = 1
        self.capacity = 100
        self.storage_state = 0


    def step(self, charge_power_in, step_index):
        charge_power_out = charge_power_in
        return charge_power_out



    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state
