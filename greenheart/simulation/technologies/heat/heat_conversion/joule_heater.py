

class JouleHeater:
    def __init__(self):
        self.eta = 0.99
        self.state = 0


    def step(self, t, u, d):
        x = self.state
        xd = self.state_transition_function(x, u, d)
        y = self.output_function(x, u, d)

        self.store(t, u, d, x, y)

        return y

    def state_transition_function(self, x, u, d):
        xd = x + u + d
        return xd

    def output_function(self, x, u, d):
        
        y = x + u + d
        return y
    
    def store(self, t, u, d, x, y):
        pass