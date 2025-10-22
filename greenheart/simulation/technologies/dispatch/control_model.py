import numpy as np
import scipy
import casadi as ca


class ControlModel:

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

        # Always initialize the control model system as a single output system
        # Then split it by duplicating rows/columns of D or F and add the appropriate constraints

        # State space matrices
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F

        # System dimensions
        self.m = self.B.shape[1]
        self.n = self.A.shape[0]
        self.p = self.C.shape[0]
        self.o = self.E.shape[1]

        # For any elements of the state space that get replaced by nonlinear function
        self.x_linear = [True] * self.n
        self.u_linear = [True] * self.m
        self.y_linear = [True] * self.p
        self.d_linear = [True] * self.o

        self.x_li = np.arange(self.n)
        self.x_nl = np.arange(0)

        self.u_li = np.arange(self.m)
        self.u_nl = np.arange(0)
        
        self.y_li = np.arange(self.p)
        self.y_nl = np.arange(0)
        
        self.d_li = np.arange(self.o)
        self.d_nl = np.arange(0)


        # Bounds
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
                for i in range(len(self_bound)):
                    if self_bound[i] is None:
                        if "ub" in bound:
                            self_bound[i] = np.inf
                            # self_bound[i] = ca.inf
                        else:
                            self_bound[i] = -np.inf
                            # self_bound[i] = -ca.inf
                        []
                setattr(self, bound, self_bound)

        self.constraints([], [])

    def parse_bound(self, dict_bound, self_bound):
        assert len(dict_bound) == len(self_bound), "given bounds must match the system dimensions"
        return dict_bound
    
    def constraints(self, y_position, constraint_type):

        self.ycon_lb = -np.inf * np.ones(len(y_position))
        self.ycon_ub = np.inf * np.ones(len(y_position))


        # attribute to self any constraints that need to be included in the plant model to make the component model work correctly
        # Like for stoichiometric components like heat exchanger with output > 0 type constraints


        # in control model, separate the input output aspects of the statespace from constraints that look like state space

        # >=0 system
        self.C_gt = [np.zeros((0, self.n))]
        self.D_gt = [np.zeros((0, self.m))]
        self.F_gt = [np.zeros((0, self.o))]

        # =0 system
        self.C_et = [np.zeros((0, self.n))]
        self.D_et = [np.zeros((0, self.m))]
        self.F_et = [np.zeros((0, self.o))]


        for i in range(len(y_position)):

            C_temp = np.atleast_2d(self.C[y_position[i], :])
            D_temp = np.atleast_2d(self.D[y_position[i], :])
            F_temp = np.atleast_2d(self.F[y_position[i], :])

            self.C = np.delete(self.C, y_position[i], axis=0)
            self.D = np.delete(self.D, y_position[i], axis=0)
            self.F = np.delete(self.F, y_position[i], axis=0)

            self.p -= 1

            self.ycon_lb[i] = self.y_lb[y_position[i]]
            self.ycon_ub[i] = self.y_ub[y_position[i]]

            if constraint_type[i] == "greater":
                self.C_gt.append(C_temp)
                self.D_gt.append(D_temp)
                self.F_gt.append(F_temp)
            elif constraint_type[i] == "equal":
                self.C_et.append(C_temp)
                self.D_et.append(D_temp)
                self.F_et.append(F_temp)

        self.C_gt = np.concatenate(self.C_gt, axis=0)
        self.D_gt = np.concatenate(self.D_gt, axis=0)
        self.F_gt = np.concatenate(self.F_gt, axis=0)

        self.C_et = np.concatenate(self.C_et, axis=0)
        self.D_et = np.concatenate(self.D_et, axis=0)
        self.F_et = np.concatenate(self.F_et, axis=0)

        self.y_lb = np.delete(self.y_lb, y_position)
        self.y_ub = np.delete(self.y_ub, y_position)

    def set_disturbance_domain(self, domain_list):
        # Domain should be a (1,3) shape array of 1s if there is a disturbance in that 
        # domain and 0 if not. This array is used as a mask to zero out erronious inputs 
        # from the wrong domain.
        # disturbance_domain = [electricity, heat energy, inputs]

        self.disturbance_domain = np.array(domain_list)

    def set_disturbance_reshape(self, reshape_mat):
        # This permutation matrix should be (o, 3) shape array to reshape the length 3
        # disturbance edge vector into the length o disturbance vector expected by the 
        # control model state space. 
        self.disturbance_permutation = np.array(reshape_mat)

    def set_output_domain(self, domain_list):
        # Similar to disturbance domain
        # output_domain = [electricity, heat energy, inputs]
        self.output_domain = np.array(domain_list)


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

    # If splitting node

    n = 0
    m = 3
    p = 3
    o = 1

    A = np.zeros((n, n))
    B = np.zeros((n, m))
    C = np.zeros((p, n))
    D = np.zeros((p, m))
    E = np.zeros((n, o))
    F = np.zeros((p, o))

    bounds_dict = {
        "u_lb": np.array([0, 0, 0]),
        "u_ub": np.array([None, None, None]),
        "x_lb": np.array([]),
        "x_ub": np.array([]),
        "y_lb": np.array([0, 0, 0]),
        "y_ub": np.array([None, None, None]),
    }


    CM = ControlModel(A=A, B=B, C=C, D=D, E=E, F=F, bounds=bounds_dict)

    []
