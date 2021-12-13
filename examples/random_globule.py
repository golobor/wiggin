# import os
# import numpy as np

import pprint

import wiggin
import wiggin.core
import wiggin.actions.sim
import wiggin.actions.interactions
import wiggin.actions.constraints
import wiggin.actions.conformations


c = wiggin.core.SimConstructor()

c.add_action(
    wiggin.actions.sim.InitializeSimulation(
        N=20000,
        # platform='CPU'
        # GPU='1',
        error_tol=0.01,
        collision_rate=0.003,
    ),
)

c.add_action(
    wiggin.actions.interactions.Chains(
        wiggle_dist=0.25,
        repulsion_e=1.5),
)


c.add_action(
    wiggin.actions.conformations.RandomWalkConformation()
)


c.add_action(
    wiggin.actions.constraints.SphericalConfinement(
        density=1 / (2.0 ** 3)
    )
)

c.add_action(
    wiggin.actions.sim.LocalEnergyMinimization()
)

c.add_action(
    wiggin.actions.sim.BlockStep(
        num_blocks=1000,
    ),
)

c.auto_name(root_data_folder='./data/random_globule/')
c.configure()

pprint.pprint(c.action_args)
pprint.pprint(c.config)

c.save_config()
c.run()
