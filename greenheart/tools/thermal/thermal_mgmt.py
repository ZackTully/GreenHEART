import numpy as np
import matplotlib.pyplot as plt
import cProfile

from greenheart.simulation.technologies.heat.heat_exchange.heat_exchanger import (
    HeatExchanger,
)
from greenheart.simulation.technologies.heat.heat_storage.thermal_energy_storage import (
    ThermalEnergyStorage,
    EnduringLockHopper,
    EnduringPressurizedFluidizedBedHeatExchanger,
    EnduringParticleHeater,
    EnduringParticleHoist,
    EnduringParticleSilo,
    EnduringPipeline,
    EnduringGeneral,
)

from greenheart.simulation.technologies.heat.materials import Quartz, Hydrogen


def run_h2_heating(
    hopp_results,
    greenheart_config,
    electrolyzer_physics_results,
    h2_storage_results,
    verbose=False,
    simulator=None,
):

    # Based off tools.eco.electrolysis

    # Assume no grid connection for now

    h2_electrolyzer_kgphr = electrolyzer_physics_results["H2_Results"][
        "Hydrogen Hourly Production [kg/hr]"
    ]

    h2_storage_soc = h2_storage_results["hydrogen_storage_soc"]
    h2_storage_kgphr = np.diff(h2_storage_soc)
    # NOTE this might not be good in all cases:
    h2_storage_kgphr = np.concatenate(
        [[0], h2_storage_kgphr],
    )
    # h2_storage_kgphr = np.concatenate(
    #     [[np.sum(h2_storage_kgphr)], h2_storage_kgphr],
    # )

    if "energy_to_heating_kw" in hopp_results:
        energy_to_heat_kw = np.asarray(hopp_results["energy_to_heating_kw"])
    else:
        energy_to_heat_kw = np.zeros(
            len(hopp_results["combined_hybrid_power_production_hopp"])
        )

    Q_to_h2_heating_kw, P_to_h2_heating_kw, TES, sizing_results, tes_sim_results = (
        run_thermal_energy_storage(energy_to_heat_kw, simulator=simulator)
    )

    h2_heating_results = hydrogen_heating(
        Q_to_h2_heating_kw, P_to_h2_heating_kw, h2_electrolyzer_kgphr, h2_storage_kgphr
    )
    cost_results = thermal_energy_storage_costs(sizing_results, TES)


    h2_heating_results.update({"Q to h2 heating kw": Q_to_h2_heating_kw, "P to h2 heating kw": P_to_h2_heating_kw})
    for key in tes_sim_results.keys():
        h2_heating_results.update({key: tes_sim_results[key]})


    h2_heating_results.update({"TES_sizing": sizing_results, "TES_costs": cost_results})

    # plot_h2_heating_results(h2_heating_results)

    return h2_heating_results, sizing_results, cost_results, TES


def plot_h2_heating_results(h2_heating_results):

    fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")

    t = np.arange(0, len(h2_heating_results["heating_available_kwh"]), 1)

    # ax[0,0].fill_between(t, np.zeros(len(t)), h2_heating_results["heating_available_kwh"], step="post")
    ax[0, 0].fill_between(
        t, np.zeros(len(t)), h2_heating_results["heating_used_kwh"], step="post"
    )
    ax[0, 0].fill_between(
        t,
        h2_heating_results["heating_used_kwh"],
        h2_heating_results["heating_used_kwh"]
        + h2_heating_results["heating_wasted_kwh"],
        step="post",
    )

    ax[1, 0].fill_between(
        t,
        np.zeros(len(t)),
        h2_heating_results["heating_wasted_kwh"]
        / h2_heating_results["heating_available_kwh"],
        step="post"
    )

    ax[0, 1].fill_between(
        t, np.zeros(len(t)), h2_heating_results["hydrogen_heated_kgphr"], step="post"
    )
    ax[0, 1].fill_between(
        t,
        h2_heating_results["hydrogen_heated_kgphr"],
        h2_heating_results["hydrogen_heated_kgphr"]
        + h2_heating_results["hydrogen_wasted_kgphr"],
        step="post",
    )

    ax[1, 1].fill_between(
        t,
        np.zeros(len(t)),
        h2_heating_results["hydrogen_wasted_kgphr"]
        / (h2_heating_results["hydrogen_heated_kgphr"]+ h2_heating_results["hydrogen_wasted_kgphr"]),
        step="post"
    )

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_ylim([0, ax[i,j].get_ylim()[1]])
            ax[i,j].set_xlim([0, 8760])


    print(f"Total heat energy wasted [kwh]: {np.sum(h2_heating_results['heating_wasted_kwh']) : .2f}")
    print(f"Total hydrogen wasted [kg]: {np.sum(h2_heating_results['hydrogen_wasted_kgphr']) : .2f}")

    []


def thermal_energy_storage_costs(sizing_results, TES):

    capex_silo_usd = (
        EnduringParticleSilo.single_unit_capital_cost_USD * sizing_results["n_silos"]
    )
    capex_skip_hoist_usd = (
        EnduringParticleHoist.single_unit_capital_cost_USD * sizing_results["n_hoists"]
    )

    capex_PFBHX = (
        EnduringPressurizedFluidizedBedHeatExchanger.single_unit_capital_cost_USD
        * sizing_results["n_PFBHX"]
    )
    capex_particle_heater = (
        EnduringParticleHeater.single_unit_capital_cost_USD
        * sizing_results["n_heaters"]
    )
    capex_lockhopper = (
        EnduringLockHopper.single_unit_capital_cost_USD * sizing_results["n_hoppers"]
    )

    Q_tes_kwhth = (
        EnduringParticleSilo.single_unit_capacity_GWhth
        * 1e6
        * sizing_results["n_silos"]
    )

    cost_results = {
        "capex_tes_usd": capex_silo_usd
        + capex_skip_hoist_usd
        + capex_PFBHX
        + capex_particle_heater
        + capex_lockhopper,
        "opex_tes_usdpyr": EnduringGeneral.operation_and_maintenance_usdpkwhth
        * Q_tes_kwhth,
    }

    return cost_results


def hydrogen_heating(heat_energy_kw, power_kw, h2_elec_kgphr, h2_storage_kgphr):
    h2_storage_charging = np.where(h2_storage_kgphr >= 0, h2_storage_kgphr, 0)
    h2_storage_discharging = np.where(h2_storage_kgphr < 0, -h2_storage_kgphr, 0)

    electrolyzer_to_h2heating_kgphr = h2_elec_kgphr - h2_storage_charging
    storage_to_h2heating_kgphr = h2_storage_discharging

    T_elec = 80 + 273.15  # [K]
    T_storage = 25 + 273.15  # [K]

    T_h2 = (
        T_elec * electrolyzer_to_h2heating_kgphr
        + T_storage * storage_to_h2heating_kgphr
    ) / (electrolyzer_to_h2heating_kgphr + storage_to_h2heating_kgphr)

    

    T_h2 = np.nan_to_num(T_h2)
    T_h2 = np.where(T_h2 != 0, T_h2, 298.15)

    H2 = Hydrogen()

    T_DRI = 900 + 273.15  # [K]

    H_cold_kJ = np.array([H2.H(T) for T in T_h2]) / H2.molar_mass * 1000
    # H_cold_kJ = H2.H(T_h2) / H2.molar_mass * 1000
    # H_hot_kJ = np.array([H2.H(T) for T in T_DRI]) / H2.molar_mass * 1000 * np.ones(len(T_h2))
    H_hot_kJ = H2.H(T_DRI) / H2.molar_mass * 1000 * np.ones(len(T_h2))

    h2_heating_energy_kwhpkg = (H_hot_kJ - H_cold_kJ) / 3600
    h2_heating_energy_kwh = h2_heating_energy_kwhpkg * (
        electrolyzer_to_h2heating_kgphr + storage_to_h2heating_kgphr
    )

    # Check if the heating energy is enough
    # If it is not then do something

    heating_energy_available = heat_energy_kw + power_kw
    heating_energy_used_kwh = np.where(
        heating_energy_available >= h2_heating_energy_kwh,
        h2_heating_energy_kwh,
        heating_energy_available,
    )
    heating_energy_deficit_kwh = np.where(
        heating_energy_available < h2_heating_energy_kwh,
        h2_heating_energy_kwh - heating_energy_available,
        0,
    )
    hydrogen_heated_kg = (1 - heating_energy_deficit_kwh / h2_heating_energy_kwh) * (
        electrolyzer_to_h2heating_kgphr + storage_to_h2heating_kgphr
    )
    hydrogen_wasted_kg = (
        electrolyzer_to_h2heating_kgphr + storage_to_h2heating_kgphr
    ) - hydrogen_heated_kg
    heating_energy_wasted_kwh = heating_energy_available - heating_energy_used_kwh

    h2_heating_results = {
        "heating_available_kwh": heating_energy_available,
        "heating_used_kwh": heating_energy_used_kwh,
        "heating_deficit_kwh": heating_energy_deficit_kwh,
        "heating_wasted_kwh": heating_energy_wasted_kwh,
        "hydrogen_heated_kgphr": hydrogen_heated_kg,
        "hydrogen_wasted_kgphr": hydrogen_wasted_kg,
    }

    return h2_heating_results


def run_thermal_energy_storage(energy_to_heat_kw, simulator=None):

    if simulator is None:

        energy_to_heat_mean = np.mean(energy_to_heat_kw)

        tes_charging_kw = energy_to_heat_kw - energy_to_heat_mean
        
        tes_max_charging_kw = np.max(tes_charging_kw)
        tes_min_charging_kw = np.min(tes_charging_kw)

        tes_charge_kwh = np.cumsum(tes_charging_kw)
        tes_charge_kwh -= np.min(tes_charge_kwh)

        # correct for heat loss
        tes_charging_kw += np.mean(tes_charge_kwh * (0.01 / 24))


        tes_max_charge_kwh = np.max(tes_charge_kwh)
        tes_min_charge_kwh = np.min(tes_charge_kwh)
    else:
        rts_tes = simulator.models["thermal_energy_storage"]
        tes_min_charging_kw = rts_tes.max_charge_kWhphr
        tes_max_charging_kw = rts_tes.max_discharge_kWhphr
        tes_min_charge_kwh = rts_tes.H_capacity_kWh
        tes_max_charge_kwh = 0

        tes_charging_kw = np.stack([simulator.models["thermal_energy_storage"].P_used_store , simulator.models["thermal_energy_storage"].Q_out_store ]).T

    sizing_results = size_thermal_energy_storage(
        tes_min_charging_kw, tes_max_charging_kw, tes_min_charge_kwh, tes_max_charge_kwh
    )


    # cProfile.run("simulate_thermal_energy_storage(sizing_results, energy_to_heat_kw, tes_charging_kw)", "/Users/ztully/Documents/hybrids_code/GH_scripts/profiling_test/profiles/TES_sim")
    # cProfile.run("Q_to_h2_heating_kw, P_to_h2_heating_kw, TES = simulate_thermal_energy_storage(sizing_results, energy_to_heat_kw, tes_charging_kw)", "/Users/ztully/Documents/hybrids_code/GH_scripts/profiling_test/profiles/TES_sim")

    Q_to_h2_heating_kw, P_to_h2_heating_kw, tes_sim_results, TES = simulate_thermal_energy_storage(
        sizing_results, energy_to_heat_kw, tes_charging_kw, simulator
    )

    # import cProfile
    # cProfile.runctx("simulate_thermal_energy_storage(sizing_results, energy_to_heat_kw, tes_charging_kw)", globals(), locals(),  "/Users/ztully/Documents/hybrids_code/GH_scripts/profiling_test/profiles/TES_sim")
    return Q_to_h2_heating_kw, P_to_h2_heating_kw, TES, sizing_results, tes_sim_results


def size_thermal_energy_storage(min_kw, max_kw, min_kwh, max_kwh):

    dt = 1  # [hr] assumption NOTE double check this

    T_cold = 300 + 273.15  # [K]
    T_hot = 1200 + 273.15  # [K]

    sand = Quartz()
    Q_charging_heating_kJpkg = (sand.H(T_hot) - sand.H(T_cold)) / sand.molar_mass * 1000
    Q_charging_heating_kwhpkg = Q_charging_heating_kJpkg / 3600
    Q_discharging_heating_kJpkg = (
        (sand.H(T_cold) - sand.H(T_hot)) / sand.molar_mass * 1000
    )
    Q_discharging_heating_kwhpkg = Q_discharging_heating_kJpkg / 3600

    sand_kg = max_kwh / Q_charging_heating_kwhpkg

    # capacity has to do with number of silos
    storage_capacity_GWhe = (max_kwh - min_kwh) / 1e6

    n_silos_frac = (
        storage_capacity_GWhe / EnduringParticleSilo.single_unit_capacity_GWhth
    )
    n_silos = np.ceil(
        storage_capacity_GWhe / EnduringParticleSilo.single_unit_capacity_GWhth
    )

    # charging has to do with enduring particle heater and particle hoist
    eta_electric_heating = 0.98

    max_charging_power_mw = max_kw / 1e3 / eta_electric_heating
    max_charging_energy_kwh = max_kw / dt / eta_electric_heating

    n_heaters_frac = (
        max_charging_power_mw / EnduringParticleHeater.single_unit_capacity_MW
    )
    n_heaters = np.ceil(
        max_charging_power_mw / EnduringParticleHeater.single_unit_capacity_MW
    )

    charging_sand_flow_rate_kgph = max_charging_energy_kwh / Q_charging_heating_kwhpkg

    EPH = EnduringParticleHoist()
    hoist_rated_sand_kgph = (EPH.particle_load_per_lift_kg / EPH.lift_time_s) * 3600
    # hoist_rated_sand_kgph = (3600 / EnduringParticleHoist.lift_time_s) * EnduringParticleHoist.particle_load_per_lift_kg

    # Naive approach maybe
    n_hoists_frac = charging_sand_flow_rate_kgph / hoist_rated_sand_kgph
    n_hoists = np.ceil(charging_sand_flow_rate_kgph / hoist_rated_sand_kgph)

    # discharging has to do with enduring lock hopper and enduring particlefluiedizedbed
    # page 67 - need a skip hoist for discharging too

    max_discharging_power_mw = -min_kw / 1e3
    max_discharging_energy_kwh = -min_kw / dt

    h2 = Hydrogen()
    # oversize assume all heating is coming from cold hydrogen storage at T = 20 C
    T_h2_h2s = 25 + 273.15  # [K]
    T_h2_pfbin = T_cold  # [K]
    T_h2_pfbout = T_hot

    # Heating hydrogen from 20 C to PFBHX inlet
    P_heating_h2_kJpkg = (h2.H(T_h2_pfbin) - h2.H(T_h2_h2s)) / h2.molar_mass * 1000
    P_heating_h2_kwhpkg = P_heating_h2_kJpkg / 3600

    Q_heating_h2_kJpkg = (h2.H(T_h2_pfbout) - h2.H(T_h2_pfbin)) / h2.molar_mass * 1000
    Q_heating_h2_kwhpkg = Q_heating_h2_kJpkg / 3600

    # Cooling sand from hot silo temperature to cold silo temperature
    discharging_sand_flow_rate_kgph = (
        -max_discharging_energy_kwh / Q_discharging_heating_kwhpkg
    )

    ELH = EnduringLockHopper()
    lock_hopper_sand_flow_rate_kgph = (
        ELH.particle_load_per_hopper_kg / (ELH.charge_time_s + ELH.discharge_time_s)
    ) * 3600

    n_hoppers_frac = discharging_sand_flow_rate_kgph / lock_hopper_sand_flow_rate_kgph
    n_hoppers = np.ceil(n_hoppers_frac)

    # Don't know how to size the PFBHX - come back to this later
    # assume ENDURING system with 1 PFBHX is designed for 135 MWe turbine
    # assume RTE = 50% comes from the gas turbine cycle since charging is 98% efficient
    # 135 / .5 = 270 MWth is the equivalent rating for the the heating output from the PFBHX
    PFBHX_rating_kwth = 135 / 0.5 * 1e3
    PFBHX_rating_kwhth = PFBHX_rating_kwth / dt

    n_PFBHX_frac = -min_kw / PFBHX_rating_kwth
    n_PFBHX = np.ceil(n_PFBHX_frac)

    sizing_results = {
        "T_sand_cold_K": T_cold,
        "T_sand_hot_K": T_hot,
        "Q_sand_heating_kwhpkg": Q_charging_heating_kwhpkg,
        "Q_sand_cooling_kwhpkg": Q_discharging_heating_kwhpkg,
        "sand_capacity_kg": sand_kg,
        "n_silos": n_silos,
        "n_silos_fraction": n_silos_frac,
        "eta_electric_heating": eta_electric_heating,
        "n_heaters": n_heaters,
        "n_heaters_fraction": n_heaters_frac,
        "n_hoists": n_hoists,
        "n_hoists_fraction": n_hoists_frac,
        "hoist_rated_massflow": hoist_rated_sand_kgph,
        "n_hoppers": n_hoppers,
        "n_hoppers_fraction": n_hoppers_frac,
        "hopper_rated_massflow": lock_hopper_sand_flow_rate_kgph,
        "n_PFBHX": n_PFBHX,
        "n_PFBHX_fraction": n_PFBHX_frac,
    }

    return sizing_results


def simulate_thermal_energy_storage(sizing_results, energy_to_heat_kw, tes_charging_kw, simulator = None):

    # Initialize thermal energy storage from sizing results
    if simulator is None: 
        tes_config = create_thermal_energy_storage_config(sizing_results)
        TES = ThermalEnergyStorage(**tes_config)
        tes_charge = np.where(tes_charging_kw >= 0, tes_charging_kw, 0)
        tes_discharge = np.where(tes_charging_kw < 0, -tes_charging_kw, 0)

        Q_out_store = np.zeros(len(energy_to_heat_kw))

        import time

        t0 = time.time()

        for step_index in range(len(energy_to_heat_kw)):

            Q_out, P_passthrough, P_curtail = TES.step(
                available_power=energy_to_heat_kw[step_index, None],
                # dispatch=tes_charging_kw[step_index, None],
                dispatch=np.array([tes_charge[step_index], tes_discharge[step_index]]),
                step_index=step_index,
            )
            Q_out_store[step_index] = Q_out

        duration = time.time() - t0

        Q_to_h2_heating_kwh = Q_out_store
        # TODO Double check on this - It shouldnt take any electrical power
        print("Did you remove electrical heating option?")
        P_to_h2_heating_kwh = energy_to_heat_kw - TES.P_used_store # / 3600

    else:
        TES = simulator.models["thermal_energy_storage"]
        Q_to_h2_heating_kwh = TES.Q_out_store
        P_to_h2_heating_kwh = np.zeros(Q_to_h2_heating_kwh.shape)


        tes_charge = TES.P_used_store
        tes_discharge = TES.Q_out_store


    tes_sim_results = {
        "tes_soc": TES.SOC_store,
        "tes_charging_kwh": tes_charge, 
        "tes_discharging_kwh": tes_discharge
    }


    return Q_to_h2_heating_kwh, P_to_h2_heating_kwh, tes_sim_results, TES


def create_thermal_energy_storage_config(sizing_results):

    # TODO double check does it make sense to treat hot and cold silos completely separately
    M_hot_capacity_kg = (
        sizing_results["n_silos"] * EnduringParticleSilo.single_unit_capacity_kg
    )
    M_cold_capacity_kg = M_hot_capacity_kg

    max_sand_flow_charging_kgph = (
        sizing_results["n_hoists"] * sizing_results["hoist_rated_massflow"]
    )
    max_sand_flow_discharging_kgph = (
        sizing_results["n_hoppers"] * sizing_results["hopper_rated_massflow"]
    )

    tes_config = {
        "M_hot_capacity": M_hot_capacity_kg,
        "M_buffer_capacity": M_cold_capacity_kg,
        "mdot_max_charge": max_sand_flow_charging_kgph,
        "mdot_max_discharge": max_sand_flow_discharging_kgph,
        "T_hot_target": sizing_results["T_sand_hot_K"] - 273.15,
        "T_buffer_target": sizing_results["T_sand_cold_K"] - 273.15,
        "initial_SOC": 0.5,
        "M_total": sizing_results["sand_capacity_kg"],
    }

    return tes_config
