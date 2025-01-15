from greenheart.simulation.technologies.heat.storage_media import StorageParticle,StorageModel,HeatedGas
import numpy as np
def heat_hydrogen(h2_from_electrolyzer_to_dri,h2_out_of_storage):
    
    mH2_S2DRI = np.array(h2_out_of_storage) #hydrogen mass: storage to DRI
    # mH2_E2S = [] #hydrogen mass: electrolyzer into storage
    mH2_E2DRI = np.array(h2_from_electrolyzer_to_dri) #hydrogen mass: electrolyzer to DRI

    particle_config = {
        "heat capacity":1155,
        "thermal conductivity":0.7,
        "melting temperature":1710,
        "particle name":"silica sand"
        }
    tes_config = {
        "storage temperature":1200,
        "cooled particle inlet temperature":300
        }
    gas_config = {
        "name":"H2",
        "heat capacity":14.304*1e3,
        "target temperature":900,
        "electrolyzer temperature":80,
        "gas name":"H2",
        "molecular weight": 2.016
        }

    particle = StorageParticle(particle_config=particle_config)
    tes = StorageModel(particle,tes_config = tes_config)
    gas = HeatedGas(gas_config=gas_config)

    
    dt = 3600
    tH2_hot_DRI = 900 #hydrogen inlet temp for DRI
    tH2_E = 80 #hydrogen outlet temp from electrolyzer
    tH2_S = 20 #hydrogen outlet temp from storage

    tP_S2H2 = 1200 #particle outlet temp from storage to heat hydrogen
    tP_OnDeck = 300 #particle temp of on-deck ready to heat and store 

    #total hydrogen mass to be heated [kg]
    mH2_DRI = mH2_S2DRI + mH2_E2DRI 
    #temperature of hydrogen from storage and hydrogen from electrolyzer to be heated
    tH2_cold_DRI = ((gas.Cp*mH2_S2DRI*tH2_S) + (gas.Cp*mH2_E2DRI*tH2_E))/(gas.Cp*mH2_S2DRI + gas.Cp*mH2_E2DRI)
    # mass of particles needed to heat hydrogen
    dT_H2 = tH2_hot_DRI - tH2_cold_DRI #change in hydrogen temp 
    dT_P = tP_S2H2 - tH2_hot_DRI #change in particle temp
    # mass of particles needed to heat hydrogen
    # mP_S2H2 = (mH2_DRI*gas.Cp*dT_H2)/(-1*particle.Cp*dT_P)
    mP_S2H2 = (mH2_DRI*gas.Cp*dT_H2)/(particle.Cp*dT_P)
    
    # change in temperature to heat particles being put into storage
    dT_Pcool = tP_S2H2 - tP_OnDeck
    energy_needed_to_heat_particles = mP_S2H2*particle.Cp*dT_Pcool #J
    energy_needed_to_heat_particles_kWh = energy_needed_to_heat_particles/(dt*1e3)
    return mP_S2H2,energy_needed_to_heat_particles_kWh #particle mass demand
#calculate storage capacity of TES based on particle mass

# put particles used
#tP_reserved_silo = ((particle.Cp*mP_S2H2*tH2_hot_DRI) + (particle.Cp*mH2_E2DRI*tH2_E))/(particle.Cp*mH2_S2DRI + particle.Cp*mH2_E2DRI)