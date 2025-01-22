# naive model - just use heat capacity time mass

import numpy as np

from attrs import define, field
from typing import Optional
from hopp.type_dec import FromDictMixin


@define
class StorageParticle(FromDictMixin):
    # From ESG model
    Cp: Optional[float] = field(default=1155.0)  # J/kg-K
    k: Optional[float] = field(default=0.7)  # W/m-K
    T_melt: Optional[int] = field(default=1710)  # C
    name: Optional[str] = field(default="silica sand")


@define
class H2Gas(FromDictMixin):
    # From ESG model
    Cp: Optional[float] = field(default=14304)  # J/kg-K
    M_gas: Optional[float] = field(default=2.016)  # grams/mol
    name: Optional[str] = field(default="H2")
    T_final: Optional[float] = field(default=900)  # deg C

@define
class Temperatures(FromDictMixin):
    T_H2_DRI: Optional[float] = field(default=900) # [C] hydrogen input to DRI
    T_H2_electrolyzer: Optional[float] = field(default=80) # [C] hydrogen outlet from electrolyzer
    T_H2_storage: Optional[float] = field(default=20) # [C] hydrogen outlet temperature from storage

    T_P_storage: Optional[float] = field(default=1200) # [C] particle outlet temperature from storage
    T_P_ondeck: Optional[float] = field(default=300) # [C] particle temperature on-deck ready to heat and store NOTE: Is this 300 K?


class HeatExchanger:
    def __init__(self):
        self.particle_properties = StorageParticle()
        self.hydrogen_properties = H2Gas()

    def input_output(
        self, mdot_h2, T_h2_in, T_h2_out, mdot_particle, T_particle_in, T_particle_out
    ):
        # mdot in kg / s

        assert T_h2_out < T_particle_in, "heat wont transfer to hydrogen otherwise"

        # Assume no heat is lost in the heat exchange
        Qdot_to_hydrogen = self.hydrogen_properties.Cp * mdot_h2 * (T_h2_out - T_h2_in)
        Qdot_from_particle = -Qdot_to_hydrogen
        T_particle_out = (
            Qdot_from_particle / (self.particle_properties.Cp * mdot_particle)
            + T_particle_in
        )

        assert T_particle_out > -273 , "can be below absolute zero"

        return Qdot_to_hydrogen, T_particle_out


    def step(self, input, desired_output, step_index):
        return np.mean([input, desired_output])


if __name__ == "__main__":
    HX = HeatExchanger()

    Qdot, T_particle_out = HX.input_output(
        mdot_h2=100 / 3600,
        T_h2_in=25,
        T_h2_out=900,
        mdot_particle=1000 / 3600,
        T_particle_in=1500,
        T_particle_out=10,
    )

    []
