from subprocess import run
from pathlib import Path
import graphviz
import cantera as ct

gas = ct.Solution('gri30.yaml')
gas.TPX = 1300.0, ct.one_atm, 'CO:0.4, O2:1'
r = ct.IdealGasReactor(gas, clone=False)
net = ct.ReactorNet([r])
T = r.T
while T < 1900:
    net.step()
    T = r.T
    
element = 'O'

diagram = ct.ReactionPathDiagram(gas, element)
diagram.title = 'Reaction path diagram following {0}'.format(element)
diagram.label_threshold = 0.01

dot_file = 'rxnpath.dot'
img_file = 'rxnpath.png'
img_path = Path.cwd().joinpath(img_file)

diagram.write_dot(dot_file)
print(diagram.get_data())

print(f"Wrote graphviz input file to '{Path.cwd().joinpath(dot_file)}'.")

run(f"dot {dot_file} -Tpng -o{img_file} -Gdpi=200".split())
print(f"Wrote graphviz output file to '{img_path}'.")

