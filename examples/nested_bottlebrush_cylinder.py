import os
# import numpy as np

import wiggin.simconstructor as smc
import wiggin.mitosimconstructor as msmc

c = smc.SimConstructor()

c.add_action(
    smc.InitializeSimulation(
        N=200*4*500,
        # platform='CPU'
        # GPU='0',
        error_tol=0.01,
        collision_rate=0.003,
#        max_Ek=1000,
    ),
)

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
    msmc.AddDynamicCylinderCompression(
        powerlaw=2,
        initial_block = 1,
        final_block = 300,
        final_axial_compression = 2
    )
)

c.add_action(
    smc.LocalEnergyMinimization(tolerance=10)
)

c.add_action(
    smc.BlockStep(
        num_blocks=30000,
#        block_size=10000
    ),
)

c.auto_name(root_data_folder='./data/nested_bottlebrush_cylinder/')

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
