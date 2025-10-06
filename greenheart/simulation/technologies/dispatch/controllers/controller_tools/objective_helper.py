import numpy as np
import casadi as ca


class Objective:
    def __init__(
        self,
        mpc,
        horizon: int,
        active_terms: list[str],
        weights: list[float],
        references: dict,
        capacities: dict,
        
        # var_inds: dict,
    ):

        self.mpc = mpc

        self.horizon = horizon

        self.active_terms = active_terms

        self.weights = weights

        self.steel_ref = references["steel"]

        self.x_bes_ref = references["x_bes"]
        self.x_tes_ref = references["x_tes"]
        self.x_h2s_ref = references["x_h2s"]

        self.soc_bes_ref = references["soc_bes"]
        self.soc_tes_ref = references["soc_tes"]
        self.soc_h2s_ref = references["soc_h2s"]

        self.x_bes_max = capacities["x_bes_max"]
        self.x_tes_max = capacities["x_tes_max"]
        self.x_h2s_max = capacities["x_h2s_max"]
        self.x_bes_min = capacities["x_bes_min"]
        self.x_tes_min = capacities["x_tes_min"]
        self.x_h2s_min = capacities["x_h2s_min"]

        self.var_inds = self.get_objective_var_inds()

        self.all_terms = [
            "output_tracking",
            "gridcurtail",
            "bes_simultaneous",
            "tes_simultaneous",
            "h2s_simultaneous",
            "bes_state",
            "tes_state",
            "h2s_state",
            "bes_soc_state",
            "tes_soc_state",
            "h2s_soc_state",
            "bes_terminal",
            "tes_terminal",
            "h2s_terminal",
            # "storage_state_LQ>'
        ]

        self.inactive_terms = [t for t in self.all_terms if t not in self.active_terms]

        assert all(
            [(t in self.weights) for t in self.active_terms]
        ), f"These objective terms were activated but not given any weights: {[t for t in self.active_terms if (t not in self.weights)]}"

        for term in self.all_terms:
            if term not in self.weights:
                self.weights[term] = 1

        self.step_term_list = []
        self.non_step_term_list = []

        self.term_map = dict(
            output_tracking=self.term_step_output_tracking,
            gridcurtail=self.term_step_grid_curtail,
            bes_simultaneous=self.term_step_bes_simultaneous,
            tes_simultaneous=self.term_step_tes_simultaneous,
            h2s_simultaneous=self.term_step_h2s_simultaneous,
            bes_state=self.term_step_bes_state,
            tes_state=self.term_step_tes_state,
            h2s_state=self.term_step_h2s_state,
            bes_terminal=self.term_bes_terminal,
            tes_terminal=self.term_tes_terminal,
            h2s_terminal=self.term_h2s_terminal,
            bes_soc_state=self.term_step_bes_state_soc_quadratic,
            tes_soc_state=self.term_step_tes_state_soc_quadratic,
            h2s_soc_state=self.term_step_h2s_state_soc_quadratic,
        )

    def construct_objective(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        kwargs = dict(
            uct_var=uct_var,
            usp_var=usp_var,
            x_var=x_var,
            yex_var=yex_var,
            yco_var=yco_var,
            gridcurtail=gridcurtail,
        )

        obj_uw = 0
        obj_w = 0

        obj_terms_uw_perstep = {}
        
        obj_terms_uw = {}
        obj_terms_w = {}

        # for term in self.active_terms:
        for term in self.all_terms:
            obj_term, term_traj = self.term_map[term](**kwargs)


            obj_terms_uw.update({term: obj_term})
            obj_terms_w.update({term: self.weights[term] * obj_term})
            obj_terms_uw_perstep.update({term: term_traj})


            if term in self.active_terms:
                obj_w += self.weights[term] * obj_term

        obj_terms_uw.update({"objective": obj_w})
        obj_terms_w.update({"objective": obj_w})

        self.obj_terms_w = obj_terms_w
        self.obj_terms_uw = obj_terms_uw
        self.obj_terms_uw_traj = obj_terms_uw_perstep

        return obj_w

    def term_step_output_tracking(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0

        scaling = 1 / (self.steel_ref)**2

        obj_traj = []
        for k in range(self.horizon):
            obj_k = scaling * (self.steel_ref - yex_var[:, k]) ** 2
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_grid_curtail(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0
        pv_cap = 800000  # kW
        wind_cap = 155 * 6000  # kW
        gen_cap = pv_cap + wind_cap
        scaling = 1 / gen_cap**2
        obj_traj = []
        for k in range(self.horizon):
            obj_k = scaling * (gridcurtail[:, k]) ** 2
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_bes_simultaneous(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0

        scaling = 1 / (self.mpc.bounds_verbose["battery"]["u_ub"][0] * self.mpc.bounds_verbose["battery"]["u_ub"][1])

        obj_traj = []
        for k in range(self.horizon):
            obj_k = scaling * (
                uct_var[self.var_inds["uct_charge_bes"], k]
                * uct_var[self.var_inds["uct_discharge_bes"], k]
            )
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_tes_simultaneous(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0

        scaling = 1 / (self.mpc.bounds_verbose["thermal_energy_storage"]["u_ub"][0] * self.mpc.bounds_verbose["thermal_energy_storage"]["u_ub"][1])


        obj_traj = []
        for k in range(self.horizon):
            obj_k = scaling * (
                uct_var[self.var_inds["uct_charge_tes"], k]
                * uct_var[self.var_inds["uct_discharge_tes"], k]
            )
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_h2s_simultaneous(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0
        scaling = 1 / (self.mpc.bounds_verbose["hydrogen_storage"]["u_ub"][0] * self.mpc.bounds_verbose["hydrogen_storage"]["u_ub"][1])

        obj_traj = []
        for k in range(self.horizon):
            obj_k = scaling * (
                uct_var[self.var_inds["uct_charge_h2s"], k]
                * uct_var[self.var_inds["uct_discharge_h2s"], k]
            )
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_bes_state(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0
        obj_traj = []
        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_bes"], k] - self.x_bes_ref) ** 2
            obj_traj.append(obj_k)
            obj += obj_k

        return obj, obj_traj

    def term_step_tes_state(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0
        obj_traj = []
        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_tes"], k] - self.x_tes_ref) ** 2
            obj += obj_k
            obj_traj.append(obj_k)
        return obj, obj_traj

    def term_step_h2s_state(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        obj = 0
        obj_traj = []
        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_h2s"], k] - self.x_h2s_ref) ** 2
            obj += obj_k
            obj_traj.append(obj_k)
        return obj, obj_traj

    def term_bes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute state reference terminal
        # obj = (x_var[self.var_inds["x_bes"], self.horizon] - self.x_bes_ref)**2

        # Relative or SOC reference
        obj = (
            x_var[self.var_inds["x_bes"], self.horizon] / self.x_bes_max
            - self.soc_bes_ref
        ) ** 2

        return obj, None

    def term_tes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute reference
        # obj = (x_var[self.var_inds["x_tes"], self.horizon] - self.x_tes_ref)**2

        # Relative reference
        obj = (
            x_var[self.var_inds["x_tes"], self.horizon] / self.x_tes_max
            - self.soc_tes_ref
        ) ** 2

        return obj, None

    def term_h2s_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute reference
        # obj = (x_var[self.var_inds["x_h2s"], self.horizon] - self.x_h2s_ref)**2

        # Relative SOC reference
        obj = (
            x_var[self.var_inds["x_h2s"], self.horizon] / self.x_h2s_max
            - self.soc_h2s_ref
        ) ** 2

        return obj, None

    def term_step_storage_state_linear_quadratic(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        x_bar = np.array([[self.x_bes_max, self.x_tes_max, self.x_h2s_max]])

        Q_quad = 3 * np.eye(3) - np.ones((3, 3))
        Q_lin = -np.ones((1, 3))

        x_soc = x_var / x_bar

        term_value = x_soc.T @ Q_quad @ x_soc + Q_lin @ x_soc

        return term_value

    def term_step_bes_state_linear(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        pass

    def term_step_bes_state_soc_quadratic(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        scaling = 1 / ( (self.x_bes_max - self.x_bes_min) ** 2)
        term_obj = 0
        obj_traj = []
        for k in range(self.horizon):

            # obj_k = scaling * (self.x_bes_max - x_var[self.var_inds["x_bes"], k]) ** 2

            obj_k = (self.soc_bes_ref - x_var[self.var_inds["x_bes"], k] / self.x_bes_max)**2


            
            term_obj += obj_k
            obj_traj.append(obj_k)
        return term_obj, obj_traj

    def term_step_tes_state_soc_quadratic(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        scaling = 1 / ((self.x_tes_max - self.x_tes_min) ** 2)
        term_obj = 0
        obj_traj = []
        for k in range(self.horizon):
            # obj_k = scaling * (self.x_tes_max - x_var[self.var_inds["x_tes"], k]) ** 2
            obj_k = (self.soc_tes_ref - x_var[self.var_inds["x_tes"], k]/ self.x_tes_max)**2

            term_obj += obj_k
            obj_traj.append(obj_k)

        return term_obj, obj_traj

    def term_step_h2s_state_soc_quadratic(
        self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail
    ):
        scaling = 1 / ( (self.x_h2s_max - self.x_h2s_min) ** 2)
        term_obj = 0
        obj_traj = []
        for k in range(self.horizon):
            # obj_k = scaling * (self.x_h2s_max - x_var[self.var_inds["x_h2s"], k]) ** 2
            obj_k = (self.soc_h2s_ref - x_var[self.var_inds["x_h2s"], k] / self.x_h2s_max)


            term_obj += obj_k
            obj_traj.append(obj_k)
        return term_obj, obj_traj

    def get_objective_var_inds(self):
        # Set up indices for objective terms flexibly
        def getid(label, index_list):
            indices = [i for i in range(len(index_list)) if label in index_list[i]]
            assert len(indices) == 1
            return indices[0]

        mpc = self.mpc


        objective_var_inds = {}

        if "battery" in mpc.node_order:
            bes_var_inds = dict(
                uct_charge_bes=getid("uct 0 battery", mpc.mct_label),
                uct_discharge_bes=getid("uct 1 battery", mpc.mct_label),
                x_bes=getid("x 0 battery", mpc.n_label),
            )
            objective_var_inds.update(bes_var_inds)

        if "hydrogen_storage" in mpc.node_order:
            h2s_var_inds = dict(
                uct_charge_h2s=getid("uct 0 hydrogen_storage", mpc.mct_label),
                uct_discharge_h2s=getid("uct 1 hydrogen_storage", mpc.mct_label),
                x_h2s=getid("x 0 hydrogen_storage", mpc.n_label),
            )
            objective_var_inds.update(h2s_var_inds)

        if "thermal_energy_storage" in mpc.node_order:
            tes_var_inds = dict(
                uct_charge_tes=getid("uct 0 thermal_energy_storage", mpc.mct_label),
                uct_discharge_tes=getid("uct 1 thermal_energy_storage", mpc.mct_label),
                x_tes=getid("x 0 thermal_energy_storage", mpc.n_label),
            )
            objective_var_inds.update(tes_var_inds)
        return objective_var_inds

    def compare_objective_implementations(
        self, obj1, obj_terms1, obj_terms_uw1, obj2, obj_terms2, obj_terms_uw2
    ):

        ov = {
            key: np.random.rand(val.shape[0], val.shape[1])
            for key, val in self.opt_vars.items()
        }

        F_obj1 = ca.Function("obj1", list(self.opt_vars.values()), [obj1])
        F_obj2 = ca.Function("obj2", list(self.opt_vars.values()), [obj2])

        print(
            f"Objectives are probably the same: {np.float64(F_obj1(*list(ov.values()))) == np.float64(F_obj2(*list(ov.values())))}"
        )

        obj_diff = np.float64(F_obj1(*list(ov.values()))) - np.float64(
            F_obj2(*list(ov.values()))
        )

        for term in obj_terms1.keys():
            F1 = ca.Function(f"f1", list(self.opt_vars.values()), [obj_terms1[term]])
            F2 = ca.Function(f"f2", list(self.opt_vars.values()), [obj_terms2[term]])

            F1_uw = ca.Function(
                f"f1_uw", list(self.opt_vars.values()), [obj_terms_uw1[term]]
            )
            F2_uw = ca.Function(
                f"f2_uw", list(self.opt_vars.values()), [obj_terms_uw1[term]]
            )

            f_equal = np.float64(F1(*list(ov.values()))) == np.float64(
                F2(*list(ov.values()))
            )
            f_diff = np.float64(F1(*list(ov.values()))) - np.float64(
                F2(*list(ov.values()))
            )

            f_uw_equal = np.float64(F1_uw(*list(ov.values()))) == np.float64(
                F2_uw(*list(ov.values()))
            )
            f_uw_diff = np.float64(F1_uw(*list(ov.values()))) - np.float64(
                F2_uw(*list(ov.values()))
            )

            print(f"Term {term} are equal UW: {f_uw_equal}, W: {f_equal}")

            []

        pass
