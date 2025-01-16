import pandas as pd

from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import PEM_H2_Clusters as PEMClusters
from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters_STEP import PEM_H2_Clusters_Step

from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master import run_PEM_clusters


class run_PEM_clusters_step(run_PEM_clusters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def run(self, optimize=False):
        clusters = self.create_clusters()  # initialize clusters
        self.clusters = clusters
        if optimize:
            power_to_clusters = self.optimize_power_split()  # run Sanjana's code
        else:
            power_to_clusters = self.even_split_power()

        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()
        # run the step function for every value in power_to_clusters


        for step_index in range(power_to_clusters.shape[1]):
            # for ci, cluster in enumerate(clusters):
            #     cluster.step(power_to_clusters[ci, step_index], step_index)

            self.step(power_to_clusters[:, step_index], step_index)

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

    
        # self.clusters = clusters
        return h2_df_ts, h2_df_tot

    def step(self, input_power, step_index):
        for ci, cluster in enumerate(self.clusters):
            cluster.step(input_power[ci], step_index)



