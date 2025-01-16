import numpy as np
import pandas as pd

from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters_STEP import PEM_H2_Clusters_Step

from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master import run_PEM_clusters


class run_PEM_clusters_step(run_PEM_clusters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clusters = self.create_clusters()




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

    def step(self, input_power, step_index):

        # if called from run, then input power should be an array of len = n_clusters
        # if called from real-time simulation then input power should be a float
        if isinstance(input_power, float):
            input_power = self.even_split_power_step(input_power)


        total_h2_kg_hr = 0
        for ci, cluster in enumerate(self.clusters):
            h2_kg_hr = cluster.step(input_power[ci], step_index)
            total_h2_kg_hr += h2_kg_hr
        
        return total_h2_kg_hr


    def even_split_power_step(self, input_power):
        num_clusters_on = np.floor(input_power / self.cluster_min_power)

        # Saturate upper number of clusters on at the actual number of clusters
        if num_clusters_on > self.num_clusters:
            num_clusters_on = self.num_clusters

        num_clusters_on = int(num_clusters_on)

        # num_clusters_on = np.where(
        #     num_clusters_on > self.num_clusters, self.num_clusters, num_clusters_on
        # )
            

        power_per_cluster = input_power / num_clusters_on

        # power_per_cluster = [
        #     self.input_power_kw[ti] / num_clusters_on[ti]
        #     if num_clusters_on[ti] > 0
        #     else 0
        #     for ti, pwr in enumerate(self.input_power_kw)
        # ]

        # power_per_to_active_clusters = np.array(power_per_cluster)
        power_to_clusters = np.zeros( self.num_clusters)
        
        power_to_clusters[0:num_clusters_on] = power_per_cluster
        
        # for i, cluster_power in enumerate(
        #     power_per_to_active_clusters
        # ):  # np.arange(0,self.n_stacks,1):
        #     clusters_off = self.num_clusters - int(num_clusters_on[i])
        #     no_power = np.zeros(clusters_off)
        #     with_power = cluster_power * np.ones(int(num_clusters_on[i]))
        #     tot_power = np.concatenate((with_power, no_power))
        #     power_to_clusters[i] = tot_power

        # return np.transpose(power_to_clusters)

        return power_to_clusters
