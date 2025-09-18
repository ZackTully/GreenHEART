import numpy as np

class Objective:
    def __init__(self, horizon:int, active_terms:list[str], weights:list[float], references:dict, capacities:dict, var_inds:dict):

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

        self.var_inds = var_inds

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

        assert all([(t in self.weights) for t in self.active_terms]), f"These objective terms were activated but not given any weights: {[t for t in self.active_terms if (t not in self.weights)]}"

        for term in self.all_terms:
            if term not in self.weights:
                self.weights[term] = 1

        self.step_term_list = []
        self.non_step_term_list = []


        self.term_map = dict(
            output_tracking = self.term_step_output_tracking,
            gridcurtail = self.term_step_grid_curtail,
            bes_simultaneous=self.term_step_bes_simultaneous,
            tes_simultaneous=self.term_step_tes_simultaneous,
            h2s_simultaneous=self.term_step_h2s_simultaneous,
            bes_state=self.term_step_bes_state,
            tes_state=self.term_step_tes_state,
            h2s_state=self.term_step_h2s_state,
            bes_terminal = self.term_bes_terminal,
            tes_terminal = self.term_tes_terminal,
            h2s_terminal = self.term_h2s_terminal,
            bes_soc_state = self.term_step_bes_state_soc_quadratic,
            tes_soc_state = self.term_step_tes_state_soc_quadratic,
            h2s_soc_state = self.term_step_h2s_state_soc_quadratic,
        )



    def construct_objective(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        kwargs = dict(uct_var = uct_var,
                      usp_var= usp_var, 
                      x_var = x_var, 
                      yex_var = yex_var, 
                      yco_var = yco_var,
                      gridcurtail = gridcurtail)

        obj_uw = 0
        obj_w = 0

        obj_terms_uw = {}
        obj_terms_w = {}


        # for term in self.active_terms:
        for term in self.all_terms:
            obj_term = self.term_map[term](**kwargs)

            obj_terms_uw.update({term:obj_term})
            obj_terms_w.update({term:self.weights[term] * obj_term})
        

            if term in self.active_terms:
                obj_w += self.weights[term] * obj_term

        obj_terms_uw.update({"objective":obj_w})
        obj_terms_w.update({"objective":obj_w})


        self.obj_terms_w = obj_terms_w
        self.obj_terms_uw = obj_terms_uw
   



        return obj_w


    def term_step_output_tracking(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = (self.steel_ref - yex_var[:, k])**2
            obj += obj_k

        return obj



    def term_step_grid_curtail(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = (gridcurtail[:, k])**2
            obj+=obj_k

        return obj

    def term_step_bes_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = uct_var[self.var_inds["uct_charge_bes"], k] * uct_var[self.var_inds["uct_discharge_bes"], k]
            obj += obj_k

        return obj

    def term_step_tes_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0
        for k in range(self.horizon):
            obj_k = uct_var[self.var_inds["uct_charge_tes"], k] * uct_var[self.var_inds["uct_discharge_tes"], k]
            obj += obj_k

        return obj

    def term_step_h2s_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = uct_var[self.var_inds["uct_charge_h2s"], k] * uct_var[self.var_inds["uct_discharge_h2s"], k]
            obj+= obj_k

        return obj

    def term_step_bes_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_bes"], k] - self.x_bes_ref)**2
            obj += obj_k

        return obj
    
    def term_step_tes_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_tes"], k] - self.x_tes_ref)**2
            obj += obj_k

        return obj
    
    def term_step_h2s_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        obj = 0

        for k in range(self.horizon):
            obj_k = (x_var[self.var_inds["x_h2s"], k] - self.x_h2s_ref)**2
            obj += obj_k

        return obj

    def term_bes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute state reference terminal
        # obj = (x_var[self.var_inds["x_bes"], self.horizon] - self.x_bes_ref)**2
        
        # Relative or SOC reference
        obj = (x_var[self.var_inds["x_bes"], self.horizon]/self.x_bes_max - self.soc_bes_ref)**2

        
        return obj
    
    def term_tes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute reference
        # obj = (x_var[self.var_inds["x_tes"], self.horizon] - self.x_tes_ref)**2

        # Relative reference
        obj = (x_var[self.var_inds["x_tes"], self.horizon]/self.x_tes_max - self.soc_tes_ref)**2
        
        return obj
    
    def term_h2s_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

        # Absolute reference
        # obj = (x_var[self.var_inds["x_h2s"], self.horizon] - self.x_h2s_ref)**2

        # Relative SOC reference
        obj = (x_var[self.var_inds["x_h2s"], self.horizon]/self.x_h2s_max - self.soc_h2s_ref)**2
        
        return obj

    def term_step_storage_state_linear_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        x_bar = np.array([[self.x_bes_max, self.x_tes_max, self.x_h2s_max]])
        
        Q_quad = 3 * np.eye(3) - np.ones((3, 3))
        Q_lin = -np.ones((1, 3))

        x_soc = x_var / x_bar

        term_value = x_soc.T @ Q_quad @ x_soc + Q_lin @ x_soc

        return term_value


    def term_step_bes_state_linear(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        pass


    
    def term_step_bes_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        w_bes = 1 / (3 * (self.x_bes_max - self.x_bes_min)**2)
        term_obj = 0
        for k in range(self.horizon):
            term_obj += w_bes * (self.x_bes_max - x_var[self.var_inds["x_bes"], k])**2
        return term_obj
    
    def term_step_tes_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        w_tes = 1 / (3 * (self.x_tes_max - self.x_tes_min)**2)
        term_obj = 0
        for k in range(self.horizon):
            term_obj += w_tes * (self.x_tes_max - x_var[self.var_inds["x_tes"], k])**2
        return term_obj

    def term_step_h2s_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
        w_h2s = 1 / (3 * (self.x_h2s_max - self.x_h2s_min)**2)
        term_obj = 0
        for k in range(self.horizon):
            term_obj += w_h2s * (self.x_h2s_max - x_var[self.var_inds["x_h2s"], k])**2
        return term_obj
