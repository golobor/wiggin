# import os
# import numpy as np

import wiggin.simconstructor as smc
import wiggin.mitosimconstructor as msmc


c = smc.SimConstructor()


c.add_action(
    smc.InitializeSimulation(
        N=400*5*500,
        #platform='CPU'
        # GPU='0',
        # error_tol=0.01,
        # collision_rate=0.003,
        # max_Ek=1000,
    ),
)


c.add_action(
    msmc.GenerateTwoLayerLoops(
        # inner_loop_size = 400,
        outer_loop_size = 400 * 5,
        # outer_loop_spacing = 2,
    )
)


c.add_action(
    msmc.GenerateLoopBrushInitialConformation(
        helix_turn_length=int(8e6 // (200 * 400 * 5)) * 2,
        helix_step=15,
        random_loop_orientations=True,  
    ),
)


c.add_action(
    smc.SetInitialConformation(),
)


c.add_action(
    smc.AddChains(
        wiggle_dist=0.1,
        repulsion_e=1.5),
)


c.add_action(
    msmc.AddLoops(
        wiggle_dist=0.1,
    ),
)


c.add_action(
    msmc.AddBackboneTethering(),
)


c.add_action(
    msmc.AddStaticCylinderCompression(
        per_particle_volume = 1.5*1.5*1.5,
        # per_particle_volume = 1.331,
    )
)


c.add_action(
    smc.LocalEnergyMinimization(tolerance=10)
)


c.add_action(
    smc.BlockStep(
        num_blocks=1000,
#        block_size=10000
    ),
)

c.auto_name(root_data_folder = './data/')

c.add_action(
    msmc.SaveConfiguration()
)

print(c.shared_config)
print(c.action_configs)

c.configure()

print(c.shared_config)
print(c.action_configs)

c.run()
