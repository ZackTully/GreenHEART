import numpy as np

class DebugHelper:
    def __init__(self):
        pass

    def save_state_for_debug(self, x0, forecast, step_index):

        assert not self.debug_mode

        import datetime
        from pathlib import Path
        import json

        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S-%s")
        dir_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/mpc_saved_states"
        dir = f"{dir_path}/mpcstate_{datetime_string}_step{step_index}"
        # Path(dir).mkdir(parents=True, exist_ok=True)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # fpath = f"{dir}/mpc_data.json"
        fpath = f"{dir}.json"

        js_kw = dict(ensure_ascii=True)
        # js_kw = dict(ensure_ascii=True, indent=4)

        with open(fpath, "w", encoding="utf-8") as f:

            plant_SS = [
                [self.A, self.Bct, self.Bsp, self.Eex],
                [self.Cco, self.Dcoct, self.Dcosp, self.Fcoex],
                [self.Cex, self.Dexct, self.Dexsp, self.Fexex],
                [self.Cze, self.Dzect, self.Dzesp, self.Fzeex],
                [self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex],
                [self.Cet, self.Detct, self.Detsp, self.Fetex],
            ]

            save_dict = dict(
                horizon=self.horizon,
                statespace=[[mat.tolist() for mat in row] for row in plant_SS],
                x0=x0.tolist(),
                forecast=forecast.tolist(),
                step_index=step_index,
                bounds={key: self.bounds[key].tolist() for key in self.bounds.keys()},
                bounds_verbose={
                    node: {
                        key: self.bounds_verbose[node][key].tolist()
                        for key in self.bounds_verbose[node].keys()
                    }
                    for node in self.bounds_verbose.keys()
                },
                dimensions=self.dims,
                labels=self.labels,
                node_order=self.node_order,
                edge_order=self.edge_order,
                weights=self.weights,
                reference=self.reference,
                ref_bes_state=self.ref_bes_state,
                weight_bes_state=self.weight_bes_state,
                ref_h2s_state=self.ref_h2s_state,
                weight_h2s_state=self.weight_h2s_state,
                ref_tes_state=self.ref_tes_state,
                weight_tes_state=self.weight_tes_state,
                M_dco_yco=self.M_dco_yco.tolist(),
                yco_ub_ind=self.yco_ub_ind.tolist(),
                yco_ub_node_ind=self.yco_ub_node_ind.tolist(),
                cols_li=self.cols_li.tolist(),
                cols_nl=self.cols_nl.tolist(),
                rows_li=self.rows_li.tolist(),
                rows_nl=self.rows_nl.tolist(),
                block_ss=self.block_ss.tolist(),
                x_bes_max=self.G.nodes["battery"]["ionode"].model.max_capacity_kWh,
                x_tes_max = self.G.nodes["thermal_energy_storage"]["ionode"].model.H_capacity_kWh,
                x_h2s_max =self.G.nodes["hydrogen_storage"]["ionode"].model.max_capacity_kg,
                repr = self.opti.__repr__(),
                return_status = self.opti.return_status(),
                casadi_stats = self.opti.stats(),      
                opti_params = str(self.opti.value_parameters()),      
                opti_variables = str(self.opti.value_variables()),  
                s_opts = self.s_opts,
                p_opts = self.p_opts,    
            )

            if hasattr(self, "x_init"):  # then try to update initial guess
                save_dict.update(
                    # dict(
                    #     uct_init=self.uc_init.tolist(),
                    #     usp_init=self.us_init.tolist(),
                    #     x_init=self.x_init.tolist(),
                    #     yex_init=self.ys_init.tolist(),
                    # )
                    dict(
                        step_index_init = self.step_index_store[-1],
                        uc_init = self.uct_store[-1].tolist(),
                        us_init = self.usp_store[-1].tolist(),
                        x_init = self.x_store[-1].tolist(),
                        ys_init = self.yex_store[-1].tolist(),
                        curtail_init = self.curtail_store[-1].tolist()
                    )
                )

                # self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
                # self.opti.set_initial(self.opt_vars["usp"], self.us_init)
                # self.opti.set_initial(self.opt_vars["x"], self.x_init)
                # self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

            save_dict.update({"mpc_config": self.mpc_config})

            json.dump(save_dict, f, **js_kw)

        pass

    def load_state_for_debug(self, state_dict):
        self.horizon = state_dict["horizon"]

        self.bounds = {
            key: np.array(state_dict["bounds"][key], dtype=float)
            for key in state_dict["bounds"].keys()
        }
        self.bounds_verbose = {
            node: {
                key: np.array(state_dict["bounds_verbose"][node][key], dtype=float)
                for key in state_dict["bounds_verbose"][node].keys()
            }
            for node in state_dict["bounds_verbose"].keys()
        }

        for key in state_dict["labels"].keys():
            setattr(self, f"{key}_label", state_dict["labels"][key])

        for key in state_dict["dimensions"].keys():
            setattr(self, key, np.sum(state_dict["dimensions"][key]))

        self.node_order = state_dict["node_order"]
        self.edge_order = state_dict["edge_order"]

        self.reference = state_dict["reference"]
        self.weights = state_dict["weights"]

        self.ref_bes_state = state_dict["ref_bes_state"]
        self.weight_bes_state = state_dict["weight_bes_state"]
        self.ref_h2s_state = state_dict["ref_h2s_state"]
        self.weight_h2s_state = state_dict["weight_h2s_state"]
        self.ref_tes_state = state_dict["ref_tes_state"]
        self.weight_tes_state = state_dict["weight_tes_state"]

        combined_mat = state_dict["statespace"]

        out_dims = np.array([self.n, self.pco, self.pex, self.pze, self.pgt, self.pet])
        in_dims = np.array([self.n, self.mct, self.msp, self.oex])

        for i, row in enumerate(combined_mat):
            for j, mat in enumerate(row):
                if (out_dims[i] > 0) and (in_dims[j] > 0):
                    combined_mat[i][j] = np.array(combined_mat[i][j], dtype=float)
                else:
                    combined_mat[i][j] = np.zeros((out_dims[i], in_dims[j]), dtype=float)

        # combined_mat = [[np.array(mat) for mat in row] for row in combined_mat]

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.M_dco_yco = np.array(state_dict["M_dco_yco"], dtype=float)
        self.yco_ub_ind = np.array(state_dict["yco_ub_ind"], dtype=int)
        self.yco_ub_node_ind = np.array(state_dict["yco_ub_node_ind"], dtype=int)

        self.cols_li = np.array(state_dict["cols_li"])
        self.cols_nl = np.array(state_dict["cols_nl"])
        self.rows_li = np.array(state_dict["rows_li"])
        self.rows_nl = np.array(state_dict["rows_nl"])

        self.block_ss = np.array(state_dict["block_ss"], dtype=float)


        if "x_init" in state_dict:

            self.step_index_store.append(state_dict["step_index_init"])

            self.uc_init = np.array(state_dict["uc_init"], dtype=float)
            self.us_init = np.array(state_dict["us_init"], dtype=float)
            self.x_init = np.array(state_dict["x_init"], dtype=float)
            self.ys_init = np.array(state_dict["ys_init"], dtype=float)
            self.curtail_init = np.array(state_dict["curtail_init"], dtype=float)

            self.prev_success = True

        self.x_bes_max = float(state_dict["x_bes_max"])
        self.x_tes_max = float(state_dict["x_tes_max"])
        self.x_h2s_max = float(state_dict["x_h2s_max"])

        []