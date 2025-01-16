class Steel:
    def __init__(self):
        self.eta = 0.9

    def step(self, hydrogen_in, step_index):
        steel_out = self.eta * hydrogen_in
        return steel_out