class Electrolyzer:
    def __init__(self):
        self.eta = 0.75

    def step(self, power_in):
        hydrogen_out = self.eta * power_in
        return hydrogen_out