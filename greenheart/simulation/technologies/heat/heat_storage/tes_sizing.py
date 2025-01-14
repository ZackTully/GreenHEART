import numpy as np

def simple_size_thermal_energy_storage(hot_particle_production_profile,hot_particle_demand_kg_pr_hr):
    if isinstance(hot_particle_demand_kg_pr_hr,float):
        if hot_particle_demand_kg_pr_hr<np.mean(hot_particle_production_profile):
            #make less hydrogen than demand
            hot_particle_demand_kg_pr_hr = np.mean(hot_particle_production_profile)
    else:
        if np.sum(hot_particle_demand_kg_pr_hr)<np.sum(hot_particle_production_profile):
            #make less annual hydrogen than needed
            #normalize hydrogen demand profile to be met by hydrogen production
            hot_particle_demand_kg_pr_hr = hot_particle_demand_kg_pr_hr*(np.sum(hot_particle_production_profile)/np.sum(hot_particle_demand_kg_pr_hr))
    hot_particle_diff_kg = hot_particle_production_profile - hot_particle_demand_kg_pr_hr
    tes_soc = np.cumsum(hot_particle_diff_kg)
    max_hot_particle_storage_capacity_kg = np.max(tes_soc) - np.min(tes_soc)
    max_charge_discharge_rate_kg_pr_hr = np.max(np.abs(hot_particle_diff_kg))
    return max_hot_particle_storage_capacity_kg,max_charge_discharge_rate_kg_pr_hr,hot_particle_demand_kg_pr_hr