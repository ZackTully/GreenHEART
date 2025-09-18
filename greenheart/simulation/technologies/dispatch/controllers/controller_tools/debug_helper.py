import numpy as np
import datetime
from pathlib import Path
import json


class DebugHelper:
    def __init__(self, mpc):
        self.mpc = mpc
        pass

    def save_state_for_debug(self, x0, forecast, step_index):

        assert not self.mpc.debug_mode

        CM = self.mpc.control_model

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
                [CM.A, CM.Bct, CM.Bsp, CM.Eex],
                [CM.Cco, CM.Dcoct, CM.Dcosp, CM.Fcoex],
                [CM.Cex, CM.Dexct, CM.Dexsp, CM.Fexex],
                [CM.Cze, CM.Dzect, CM.Dzesp, CM.Fzeex],
                [CM.Cgt, CM.Dgtct, CM.Dgtsp, CM.Fgtex],
                [CM.Cet, CM.Detct, CM.Detsp, CM.Fetex],
            ]

            save_dict = dict(
                horizon=self.mpc.horizon,
                statespace=[[mat.tolist() for mat in row] for row in plant_SS],
                x0=x0.tolist(),
                forecast=forecast.tolist(),
                step_index=step_index,
                bounds={key: CM.bounds[key].tolist() for key in CM.bounds.keys()},
                bounds_verbose={
                    node: {
                        key: CM.bounds_verbose[node][key].tolist()
                        for key in CM.bounds_verbose[node].keys()
                    }
                    for node in CM.bounds_verbose.keys()
                },
                dimensions=CM.dims,
                labels=CM.labels,
                node_order=self.mpc.node_order,
                edge_order=self.mpc.edge_order,
                weights=self.mpc.weights,
                reference=self.mpc.reference,
                ref_bes_state=self.mpc.ref_bes_state,
                weight_bes_state=self.mpc.weight_bes_state,
                ref_h2s_state=self.mpc.ref_h2s_state,
                weight_h2s_state=self.mpc.weight_h2s_state,
                ref_tes_state=self.mpc.ref_tes_state,
                weight_tes_state=self.mpc.weight_tes_state,
                M_dco_yco=CM.M_dco_yco.tolist(),
                yco_ub_ind=CM.yco_ub_ind.tolist(),
                yco_ub_node_ind=CM.yco_ub_node_ind.tolist(),
                cols_li=CM.cols_li.tolist(),
                cols_nl=CM.cols_nl.tolist(),
                rows_li=CM.rows_li.tolist(),
                rows_nl=CM.rows_nl.tolist(),
                block_ss=CM.block_ss.tolist(),
                # x_bes_max=self.mpc.G.nodes["battery"]["ionode"].model.max_capacity_kWh,
                # x_bes_min=self.mpc.G.nodes["battery"]["ionode"].model.min_capacity_kWh
                # x_tes_max=self.mpc.G.nodes["thermal_energy_storage"]["ionode"].model.H_capacity_kWh,
                # x_tes_min=self.mpc.x_tes_min,
                # x_h2s_max=self.mpc.G.nodes["hydrogen_storage"]["ionode"].model.max_capacity_kg,
                x_bes_max=self.mpc.x_bes_max,
                x_bes_min=self.mpc.x_bes_min,
                x_tes_max=self.mpc.x_tes_max,
                x_tes_min=self.mpc.x_tes_min,
                x_h2s_max=self.mpc.x_h2s_max,
                x_h2s_min=self.mpc.x_h2s_min,
                repr=self.mpc.opti.__repr__(),
                return_status=self.mpc.opti.return_status(),
                casadi_stats=self.mpc.opti.stats(),
                opti_params=str(self.mpc.opti.value_parameters()),
                opti_variables=str(self.mpc.opti.value_variables()),
                s_opts=self.mpc.s_opts,
                p_opts=self.mpc.p_opts,
            )

            if hasattr(self, "x_init"):  # then try to update initial guess
                save_dict.update(
                    dict(
                        step_index_init=self.mpc.step_index_store[-1],
                        uc_init=self.mpc.uct_store[-1].tolist(),
                        us_init=self.mpc.usp_store[-1].tolist(),
                        x_init=self.mpc.x_store[-1].tolist(),
                        ys_init=self.mpc.yex_store[-1].tolist(),
                        curtail_init=self.mpc.curtail_store[-1].tolist(),
                    )
                )

            save_dict.update({"mpc_config": self.mpc.mpc_config})

            json.dump(save_dict, f, **js_kw)

        pass

    def load_state_for_debug(self, state_dict):
        self.mpc.horizon = state_dict["horizon"]

        self.mpc.bounds = {
            key: np.array(state_dict["bounds"][key], dtype=float)
            for key in state_dict["bounds"].keys()
        }
        self.mpc.bounds_verbose = {
            node: {
                key: np.array(state_dict["bounds_verbose"][node][key], dtype=float)
                for key in state_dict["bounds_verbose"][node].keys()
            }
            for node in state_dict["bounds_verbose"].keys()
        }

        for key in state_dict["labels"].keys():
            setattr(self, f"{key}_label", state_dict["labels"][key])
            setattr(self.mpc, f"{key}_label", state_dict["labels"][key])

        for key in state_dict["dimensions"].keys():
            setattr(self, key, np.sum(state_dict["dimensions"][key]))
            setattr(self.mpc, key, np.sum(state_dict["dimensions"][key]))

        self.mpc.node_order = state_dict["node_order"]
        self.mpc.edge_order = state_dict["edge_order"]

        self.mpc.reference = state_dict["reference"]
        self.mpc.weights = state_dict["weights"]

        self.mpc.ref_bes_state = state_dict["ref_bes_state"]
        self.mpc.weight_bes_state = state_dict["weight_bes_state"]
        self.mpc.ref_h2s_state = state_dict["ref_h2s_state"]
        self.mpc.weight_h2s_state = state_dict["weight_h2s_state"]
        self.mpc.ref_tes_state = state_dict["ref_tes_state"]
        self.mpc.weight_tes_state = state_dict["weight_tes_state"]

        combined_mat = state_dict["statespace"]

        out_dims = np.array(
            [
                self.mpc.n,
                self.mpc.pco,
                self.mpc.pex,
                self.mpc.pze,
                self.mpc.pgt,
                self.mpc.pet,
            ]
        )
        in_dims = np.array([self.mpc.n, self.mpc.mct, self.mpc.msp, self.mpc.oex])

        for i, row in enumerate(combined_mat):
            for j, mat in enumerate(row):
                if (out_dims[i] > 0) and (in_dims[j] > 0):
                    combined_mat[i][j] = np.array(combined_mat[i][j], dtype=float)
                else:
                    combined_mat[i][j] = np.zeros(
                        (out_dims[i], in_dims[j]), dtype=float
                    )

        # combined_mat = [[np.array(mat) for mat in row] for row in combined_mat]

        CM = self.mpc.control_model

        CM.A, CM.Bct, CM.Bsp, CM.Eex = combined_mat[0]
        CM.Cco, CM.Dcoct, CM.Dcosp, CM.Fcoex = combined_mat[1]
        CM.Cex, CM.Dexct, CM.Dexsp, CM.Fexex = combined_mat[2]
        CM.Cze, CM.Dzect, CM.Dzesp, CM.Fzeex = combined_mat[3]
        CM.Cgt, CM.Dgtct, CM.Dgtsp, CM.Fgtex = combined_mat[4]
        CM.Cet, CM.Detct, CM.Detsp, CM.Fetex = combined_mat[5]

        CM.M_dco_yco = np.array(state_dict["M_dco_yco"], dtype=float)
        CM.yco_ub_ind = np.array(state_dict["yco_ub_ind"], dtype=int)
        CM.yco_ub_node_ind = np.array(state_dict["yco_ub_node_ind"], dtype=int)
        self.mpc.M_dco_yco = np.array(state_dict["M_dco_yco"], dtype=float)
        self.mpc.yco_ub_ind = np.array(state_dict["yco_ub_ind"], dtype=int)
        self.mpc.yco_ub_node_ind = np.array(state_dict["yco_ub_node_ind"], dtype=int)

        CM.cols_li = np.array(state_dict["cols_li"])
        CM.cols_nl = np.array(state_dict["cols_nl"])
        CM.rows_li = np.array(state_dict["rows_li"])
        CM.rows_nl = np.array(state_dict["rows_nl"])

        CM.block_ss = np.array(state_dict["block_ss"], dtype=float)

        if "x_init" in state_dict:

            self.mpc.step_index_store.append(state_dict["step_index_init"])

            self.mpc.uc_init = np.array(state_dict["uc_init"], dtype=float)
            self.mpc.us_init = np.array(state_dict["us_init"], dtype=float)
            self.mpc.x_init = np.array(state_dict["x_init"], dtype=float)
            self.mpc.ys_init = np.array(state_dict["ys_init"], dtype=float)
            self.mpc.curtail_init = np.array(state_dict["curtail_init"], dtype=float)

            self.mpc.prev_success = True

        self.mpc.x_bes_max = float(state_dict["x_bes_max"])
        self.mpc.x_tes_max = float(state_dict["x_tes_max"])
        self.mpc.x_h2s_max = float(state_dict["x_h2s_max"])

        self.mpc.x_bes_min = float(state_dict["x_bes_min"])
        self.mpc.x_tes_min = float(state_dict["x_tes_min"])
        self.mpc.x_h2s_min = float(state_dict["x_h2s_min"])

        []
