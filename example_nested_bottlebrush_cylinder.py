import os
import numpy as np

import simconstructor as smc
import mitosimconstructor as msmc

def nameSimulation(simconstructor, base_folder = '../data/'):
    name = []
    for action, params in simconstructor.action_params.items():
        for k,v in params.items():
            name += ['_', k, '-', str(v)]

    name = ''.join(name[1:])
    simconstructor.shared_config['name'] = name
    simconstructor.shared_config['folder'] = os.path.join(base_folder, name)


c = smc.SimulationConstructor()

c.add_action(
    smc.InitializeSimulation(
        N=200*4*500,
        #platform='CPU'
        GPU='0',
        error_tol=0.001,
        collision_rate=0.003,
#        max_Ek=1000,
    ),
)

#c.add_action(
#    msmc.GenerateSingleLayerLoops(loop_size=400),
#)
c.add_action(
    msmc.GenerateTwoLayerLoops(
        inner_loop_size = 200,
        outer_loop_size = 200 * 4,
    )
)


c.add_action(
    msmc.GenerateLoopBrushInitialConformation(),
)

c.add_action(
    smc.SetInitialConformation(),
)

c.add_action(
    msmc.AddInitConfCylindricalConfinement(),
)

c.add_action(
    smc.AddChains(
        wiggle_dist=0.05,
        repulsion_e=1.5),
)

c.add_action(
    msmc.AddLoops(
        wiggle_dist=0.05,
    ),
)

c.add_action(
    msmc.AddTipsTethering(),
)

c.add_action(
    smc.LocalEnergyMinimization(tolerance=10)
)

c.add_action(
    msmc.AddDynamicCylinderCompression(
        powerlaw=2,
        initial_block = 1,
        final_block = 100,
        final_axial_compression = 4
    )
)

c.add_action(
    smc.BlockStep(
        num_blocks=30000,
#        block_size=10000
    ),
)

c.add_action(
    smc.SaveConformation()
)


nameSimulation(c)

c.add_action(
    msmc.SaveConfiguration()
)


print(c.shared_config)
print(c.action_configs)

c.configure()

print(c.shared_config)
print(c.action_configs)
c.run()
