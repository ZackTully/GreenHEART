import numpy as np


class ControlModel:
    # linear state space with bounds model

    # state space

    # bounds

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        E: np.ndarray,
        F: np.ndarray,
        bounds:dict = {},
        discrete:bool = True
    ):

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F

        self.m = self.B.shape[1]
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self.o = self.E.shape[1]

        self.u_lb = np.array([None] * self.m)
        self.u_ub = np.array([None] * self.m)
        self.x_lb = np.array([None] * self.n)
        self.x_ub = np.array([None] * self.n)
        self.y_lb = np.array([None] * self.p)
        self.y_ub = np.array([None] * self.p)

        self.bounds_dict = bounds

        for bound in ["u_lb", "u_ub", "x_lb", "x_ub", "y_lb", "y_ub"]:
            if bound in bounds:
                self_bound = self.parse_bound(bounds[bound], getattr(self, bound))
                setattr(self, bound, self_bound)

    def parse_bound(self, dict_bound, self_bound):
        assert len(dict_bound) == len(self_bound), "given bounds must match the system dimensions"
        return dict_bound
    


if __name__ == "__main__":
    A = np.array([[0.39698364, -1.68707227], [0.06748289, 0.90310532]])
    B = np.array([[0.06748289], [0.00387579]])
    C = np.array([[0.0, 25.0]])
    D = np.array([[0.0]])
    E = np.array([[0.06748289], [0.00387579]])
    F = np.array([[0.0]])

    bounds_dict = {
        "u_lb": np.array([-5]),
        "u_ub": np.array([10]),
        "x_lb": np.array([None, None]),
        "x_ub": np.array([None, None]),
        "y_lb": np.array([-20]),
        "y_ub": np.array([20]),
    }


    CM = ControlModel(A=A, B=B, C=C, D=D, E=E, F=F, bounds=bounds_dict)

    []
