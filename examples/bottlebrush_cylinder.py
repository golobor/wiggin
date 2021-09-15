# import os
# import numpy as np

import wiggin.simconstructor as smc
import wiggin.mitosimconstructor as msmc

c = smc.SimConstructor()

c.add_action(
    smc.InitializeSimulation(
        N=20000,
        # platform='CPU'
        # GPU='1',
        error_tol=0.001,
        collision_rate=0.003,
    ),
)

c.add_action(
    msmc.GenerateSingleLayerLoops(loop_size=400),
)

c.add_action(
    msmc.GenerateLoopBrushInitialConformation(
        helix_radius=1e-9, helix_step=1e9),
)

c.add_action(
    smc.SetInitialConformation(),
)

# c.add_action(
#     msmc.AddInitConfCylindricalConfinement(),
# )

c.add_action(
    smc.AddChains(
        wiggle_dist=0.25,
        repulsion_e=1.5),
)

c.add_action(
    msmc.AddLoops(
        wiggle_dist=0.25,
    ),
)

c.add_action(
    msmc.AddTipsTethering(),
)

c.add_action(
    smc.LocalEnergyMinimization()
)

# c.add_action(
#     msmc.AddDynamicCylinderCompression(
#     powerlaw=2,
#     initial_block = 1,
#     final_block = 50,
#     final_axial_compression = 4
#     )
# )

c.add_action(
    smc.BlockStep(
        num_blocks=1000,
#        block_size=10000
    ),
)

c.auto_name(root_data_folder='./data/bottlebrush_cylinder/')

c.add_action(
    msmc.SaveConfiguration()
)


print(c.shared_config)
print(c.action_configs)

c.configure()

print(c.shared_config)
print(c.action_configs)
c.run()
