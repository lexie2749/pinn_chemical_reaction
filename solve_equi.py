import cantera as ct
g = ct.Solution('gri30.yaml')
g.TPX = 300.0, ct.one_atm, 'CH4:0.95,O2:2,N2:7.52'
g.equilibrate('TP')

g.TPX = 300.0, ct.one_atm, 'CH4:0.95,O2:2,N2:7.52'
g.equilibrate('HP')

