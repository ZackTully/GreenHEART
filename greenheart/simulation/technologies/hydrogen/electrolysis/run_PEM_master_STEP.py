import numpy as np
import pandas as pd

from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters_STEP import PEM_H2_Clusters_Step

from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master import run_PEM_clusters
from greenheart.simulation.technologies.dispatch.control_model import ControlModel


class run_PEM_clusters_step(run_PEM_clusters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clusters = self.create_clusters()

        self.max_power_kW = self.num_clusters * self.cluster_max_power

        self.create_control_model()

    def create_control_model(self):

        n = 0
        m = 0
        p = 1
        o = 1

        A = np.zeros((n, n))
        B = np.zeros((n, m))
        C = np.zeros((p, n))
        D = np.zeros((p, m))
        E = np.zeros((n, o))
        F = np.array([[1 / 46.99]]) # kg H2 / kWh
        # F = np.array([[1 / 55]]) # kg H2 / kWh

        bounds_dict = {
            "u_lb": np.array([]),
            "u_ub": np.array([]),
            "x_lb": np.array([]),
            "x_ub": np.array([]),
            "y_lb": np.array([0]),
            "y_ub": np.array([1 / 55 * self.num_clusters * self.cluster_max_power]), # NOTE rough estimate, come back and fix this
        }


        self.control_model = ControlModel(A, B, C, D, E, F)
        self.control_model.set_disturbance_domain([1, 0, 0])
        self.control_model.set_output_domain([0, 0, 1])
        self.control_model.set_disturbance_reshape(np.array([[1, 0, 0]]))




    def run(self, optimize=False):
        clusters = self.create_clusters()  # initialize clusters
        self.clusters = clusters
        if optimize:
            power_to_clusters = self.optimize_power_split()  # run Sanjana's code
        else:
            power_to_clusters = self.even_split_power()

        # run the step function for every value in power_to_clusters


        for step_index in range(power_to_clusters.shape[1]):
            # for ci, cluster in enumerate(clusters):
            #     cluster.step(power_to_clusters[ci, step_index], step_index)

            self.step(power_to_clusters[:, step_index], step_index)

        h2_df_ts, h2_df_tot = self.consolidate_sim_outcome()

        return h2_df_ts, h2_df_tot
    
    def consolidate_sim_outcome(self):

        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()

        col_names = []

        for ci, cluster in enumerate(self.clusters):
            cl_name = "Cluster #{}".format(ci)
            col_names.append(cl_name)

            # this line replaces clusters[ci].run in normal run_PEM_master
            h2_ts, h2_tot = cluster.consolidate_sim_outcome()

            h2_ts_temp = pd.Series(h2_ts, name=cl_name)
            h2_tot_temp = pd.Series(h2_tot, name=cl_name)
            if len(h2_df_tot) == 0:
                # h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
                h2_df_tot = pd.concat(
                    [h2_df_tot, h2_tot_temp], axis=0, ignore_index=False
                )
                h2_df_tot.columns = col_names

                h2_df_ts = pd.concat([h2_df_ts, h2_ts_temp], axis=0, ignore_index=False)
                h2_df_ts.columns = col_names
            else:
                # h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_tot = h2_df_tot.join(h2_tot_temp)
                h2_df_tot.columns = col_names

                h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_ts.columns = col_names

        return h2_df_ts, h2_df_tot

    def step(self, input_power, dispatch, step_index):



        if isinstance(input_power, (np.ndarray, list)):
            input_power = input_power[0]
        if isinstance(dispatch, (np.ndarray, list)):
            dispatch = dispatch[0]


        input_power, u_passthrough, u_curtail = self.low_level_controller(input_power)


        # if called from run, then input power should be an array of len = n_clusters
        # if called from real-time simulation then input power should be a float or int
        if isinstance(input_power, float) or isinstance(input_power, int):
            input_power = self.even_split_power_step(input_power)


        total_h2_kg_hr = 0
        for ci, cluster in enumerate(self.clusters):
            h2_kg_hr = cluster.step(input_power[ci], step_index)
            total_h2_kg_hr += h2_kg_hr
        
        output = total_h2_kg_hr

        return output, u_passthrough, u_curtail


    def low_level_controller(self, input_power):

        u_model = np.min([input_power, self.max_power_kW])
        u_passthrough = 0.0
        u_curtail = np.max([0, input_power - self.max_power_kW])
        return u_model, u_passthrough, u_curtail

    def even_split_power_step(self, input_power):
        num_clusters_on = np.floor(input_power / self.cluster_min_power)

        # Saturate upper number of clusters on at the actual number of clusters
        if num_clusters_on > self.num_clusters:
            num_clusters_on = self.num_clusters

        num_clusters_on = int(num_clusters_on)

        power_to_clusters = np.zeros( self.num_clusters)

        if num_clusters_on > 0:
            power_per_cluster = input_power / num_clusters_on
            power_to_clusters[0:num_clusters_on] = power_per_cluster


        return power_to_clusters
