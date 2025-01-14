from hopp.type_dec import FromDictMixin
from attrs import define, field
from typing import List, Sequence, Optional, Union
from hopp.simulation.base import BaseClass
from greenheart.simulation.technologies.hydrogen.h2_storage.storage_sizing import hydrogen_storage_capacity
from greenheart.simulation.technologies.heat.hydrogen_heating import heat_hydrogen
import numpy as np
@define
class StorageParticle(FromDictMixin):
    Cp: Optional[float] = field(default = 1155.0) #J/kg-K
    k: Optional[float] = field(default = 0.7) #W/m-K
    T_melt: Optional[int] = field(default = 1710) #C
    name: Optional[str] = field(default = "silica sand")

@define
class H2Gas(FromDictMixin):
    Cp: Optional[float] = field(default = 14304) #J/kg-K
    M_gas: Optional[float] = field(default = 2.016) #grams/mol
    name: Optional[str] = field(default = "H2")
    T_final: Optional[float] = field(default = 900) #deg C 

@define
class TESConfig(FromDictMixin):
    # particle: StorageParticle
    # gas: H2Gas
    #particle storage & outlet temp from storage to heat hydrogen
    T_storage: Optional[float] = field(default = 1200.0) #deg C
    #particle temp of on-deck ready to heat and store 
    T_buffer: Optional[float] = field(default = 300.0) #deg C
    # storage_capacity_kg: Optional[float] = field(default = None,init = False)


@define
class TESModel(BaseClass):
    particle: StorageParticle
    gas: H2Gas
    tes_config: TESConfig


    T_gas_from_source: Optional[float] = field(default = 80.0)
    T_gas_from_storage: Optional[float] = field(default = 20.0)
    dt: Optional[float] = field(default = 3600)


    def calc_temperature_of_hydrogen_delivered(self,h2_from_source,h2_from_storage):
        
        n_storage = self.gas.Cp*h2_from_storage*self.T_gas_from_storage
        n_source = self.gas.Cp*h2_from_source*self.T_gas_from_source
        d = (self.gas.Cp*h2_from_source) + (self.gas.Cp*h2_from_storage)

        #temperature of hydrogen from storage and hydrogen from electrolyzer to be heated
        T_cold_DRI = (n_storage + n_source)/d #deg C
        return T_cold_DRI
    def calc_particle_output_mass(self,h2_from_source,h2_from_storage):
        #total hydrogen mass to be heated [kg]
        h2_total_mass = h2_from_source + h2_from_storage
        #temperature of hydrogen from storage and hydrogen from electrolyzer to be heated
        T_cold_DRI = self.calc_temperature_of_hydrogen_delivered(h2_from_source,h2_from_storage)
        #change in temperature of Hydrogen
        dT_h2 = self.gas.T_final - T_cold_DRI
        #change in temperature of TES storage particles
        dT_particle = T_cold_DRI - self.tes_config.T_storage

        # mass of particles needed to heat hydrogen [kg]
        output_mass_particles_required = (h2_total_mass*self.gas.Cp*dT_h2)/(self.particle.Cp*dT_particle)
        return output_mass_particles_required

    def calc_energy_required_to_heat_particles_for_output(self,h2_from_source,h2_from_storage):
        # change in temperature to heat particles being put into storage
        output_mass_particles_required = self.calc_particle_output_mass(h2_from_source,h2_from_storage)
        dT_particle_heatup = self.tes_config.T_buffer - self.tes_config.T_storage
        energy_needed_to_heat_particles_J = output_mass_particles_required*self.particle.Cp*dT_particle_heatup #J
        energy_needed_to_heat_particles_kW = energy_needed_to_heat_particles_J/(self.dt*1e3)
        return energy_needed_to_heat_particles_kW
    
    def estimate_tes_mass_capacity(self,h2_from_source,h2_from_storage):
        output_mass_particles_required = self.calc_particle_output_mass(h2_from_source,h2_from_storage)
        tes_storage_capacity_kg = np.max(output_mass_particles_required)
        # tes_base_capacity = np.min(output_mass_particles_required)
        max_charge_discharge_mass = np.max(np.abs(np.diff(output_mass_particles_required)))
        return tes_storage_capacity_kg,max_charge_discharge_mass
    
    def estimate_tes_hourly_energy_required_bounds(self,h2_demand_kg_pr_hr):
        upper_bound_energy_kW = self.calc_energy_required_to_heat_particles_for_output(0,h2_demand_kg_pr_hr)
        lower_bound_energy_kW = self.calc_energy_required_to_heat_particles_for_output(h2_demand_kg_pr_hr,0)
        return lower_bound_energy_kW, upper_bound_energy_kW

    def open_loop_fill_tes(self,energy_input_to_particles_kW):
        energy_input_J = energy_input_to_particles_kW*(self.dt*1e3)
        dT_particle_heatup = self.tes_config.T_buffer - self.tes_config.T_storage
        mass_of_particles_input_kg = energy_input_J/(self.particle.Cp*dT_particle_heatup)
        return mass_of_particles_input_kg