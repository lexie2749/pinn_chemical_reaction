import cantera as ct
import numpy as np

gas1 = ct.Solution('gri30.yaml')
# gas1()

gas1.TP = 1200, 101325          # temperature, pressure
gas1.TD = 1200, 0.020473        # temperature, density
gas1.HP = 1.3295e7, 101325      # specific enthalpy, pressure
gas1.UV = 8.3457e6, 1/0.020473  # specific internal energy, specific volume
gas1.SP = 85222, 101325         # specific entropy, pressure
gas1.SV = 85222, 1/0.020473     # specific entropy, specific volume

print(gas1.T)
print(gas1.h)
print(gas1.UV)

gas1.X = 'CH4:1, O2:2, N2:7.52'

phi = 0.8
gas1.X = {'CH4':1, 'O2':2/phi, 'N2':2*3.76/phi}

gas1.TPX = 1200, 101325, 'CH4:1, O2:2, N2:7.52'
gas1()

gas1.X = np.ones(53)  # NumPy array of 53 ones
gas1.Y = np.ones(53)
gas1.SV = None, 2.1
gas1.TPX = None, None, 'CH4:1.0, O2:0.5'

Xmajor = gas1['CH4','O2','CO2','H2O','N2'].X
major = gas1['CH4','O2','CO2','H2O','N2']
cp_major = major.partial_molar_cp
wdot_major = major.net_production_rates
