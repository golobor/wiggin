import socket, copy, itertools, os

import numpy as np

from polychrom import simulation, forces, forcekits


class AttrDict(dict):
    @property
    def __dict__(self):
        return self

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.update(state)


class SimulationConstructor:
    def __init__(self):
        self._actions = {
            'init':[],
            'loop':[]
        }

        self._sim = None
        self.shared_config = AttrDict(
            name=None,
            N=None,
            initial_conformation=None,
            folder=None,
        )

        self.action_params = AttrDict()
        self.action_configs = AttrDict()
        pass


    def add_action(self, action):
        if action.name in self.action_params:
            raise ValueError(
                f'Action {action.name} was already added to the constructor!')
        self.action_params[action.name] = copy.deepcopy(action.params)
        self._actions[action.stage].append(action)


    def configure(self):
        for action in itertools.chain(
            self._actions['init'], self._actions['loop']):

            if action.name in self.action_configs:
                raise ValueError(
                    f'Action {action.name} has already been configured!')

            conf_res = action.configure(
                self.shared_config,
                self.action_configs)
            assert conf_res is not None, (
                    f'Action {action.name} must return two configs in .configure!')

            shared_config_added_data, action_config  = conf_res
            self.shared_config.update(shared_config_added_data) 
            self.action_configs[action.name] = action_config


    def run(self):
        for action in self._actions['init']:
            self._sim = action.run(
                    self.shared_config, 
                    self.action_configs, 
                    self._sim)

        while self._sim:
            for action in self._actions['loop']:
                self._sim = action.run(
                        self.shared_config, 
                        self.action_configs, 
                        self._sim)

                if self._sim is None:
                    break


class SimulationAction:
    _default_params = AttrDict()

    def __init__(
            self, 
            name=None,
            **kwargs
            ):
        if name is None:
            name = type(self).__name__
        self.name = name
        self.params = kwargs


    def configure(self, shared_config, action_configs):
        shared_config_added_data = AttrDict()
        action_config = AttrDict()

        for k,def_v in self._default_params.items():
            action_config[k] = self.params.get(k, def_v)

        return shared_config_added_data, action_config


    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        # TODO: rethink updating sim objects
        return sim


class InitializeSimulation(SimulationAction):
    stage = 'init'

    _default_params = AttrDict(
        N=None,
        computer_name=None,
        platform='CUDA',
        GPU='0',
        integrator = 'variableLangevin',
        error_tol = 0.001,
        mass = 1,
        collision_rate = 0.01,
        temperature = 300,
        timestep = 1.0,
        max_Ek=100,
    )


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        action_config['computer_name'] = socket.gethostname()
        shared_config_added_data['N'] = action_config['N']

        return shared_config_added_data, action_config


    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config

        self_conf = action_configs[self.name]
        sim = simulation.Simulation(
            platform=self_conf.platform,
            GPU=self_conf.GPU,
            integrator=self_conf.integrator,
            error_tol=self_conf.error_tol,
            collision_rate=self_conf.collision_rate,
            mass=self_conf.mass,
            N=shared_config.N,
            max_Ek=self_conf.max_Ek
        )  # timestep not necessary for variableLangevin

        return sim


class BlockStep(SimulationAction):
    stage = 'loop'
    _default_params = AttrDict(
        num_blocks = 100,
        block_size = int(1e4),
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if (sim.step / self_conf.block_size >= self_conf.num_blocks):
            return None
        else:
            sim.do_block(self_conf.block_size)  
            return sim


class LocalEnergyMinimization(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        max_iterations = 1000,
        tolerance = 1,
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]
        sim.local_energy_minimization(
                maxIterations=self_conf.max_iterations,
                tolerance=self_conf.tolerance
        )
        return sim
 

class AddChains(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        chains = [(0, None, 0)],
        bond_length = 1.0,
        wiggle_dist = 0.025,
        stiffness_k = None,
        repulsion_e = 2.5, ## TODO: implement np.inf 
        except_bonds = False,
    )


    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=self_conf.chains,
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength': self_conf.bond_length,
                    'bondWiggleDistance': self_conf.wiggle_dist,
                },

                angle_force_func=(
                    None if self_conf.stiffness_k is None 
                    else forces.angle_force),
                angle_force_kwargs={
                    'k': self_conf.stiffness_k 
                },

                nonbonded_force_func=forces.polynomial_repulsive,
                nonbonded_force_kwargs={
                    'trunc': self_conf.repulsion_e 
                },

                except_bonds=self_conf.except_bonds
            )
        )

        return sim


class CrosslinkParallelChains(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        chains = None,
        bond_length = 1.0,
        wiggle_dist = 0.025,
    )
    
    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if action_config['chains'] is None:
            action_config['chains'] = [((0, 
                                         shared_config_added_data['N']//2, 
                                         1),
                                        (shared_config_added_data['N']//2, 
                                         shared_config_added_data['N'], 
                                         1)
                                       ),
                                      ]

        return shared_config_added_data, action_config


    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        bonds = sum([
            zip(range(chain1[0], chain1[1], chain1[2]),
                range(chain2[0], chain2[1], chain2[2]))
            for chain1, chain2 in action_config['chains']
        ])

        sim.add_force(
            forces.harmonic_bonds(
                sim,
                bonds=bonds,
                bondLength= self_conf.bond_length,
                bondWiggleDistance= self_conf.wiggle_dist,
                name='ParallelChainsCrosslinkBonds'
            )
        )

        return sim


class SetInitialConformation(SimulationAction):
    stage = 'init'

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]
        sim.set_data(shared_config.initial_conformation)

        return sim


class AddCylindricalConfinement(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        k=0.5,
        r=None,
        top=None,
        bottom=None,
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.cylindrical_confinement(
                sim_object=sim,
                r=self_conf.r,
                top=self_conf.top,
                bottom=self_conf.bottom, 
                k=self_conf.k
            )
        )


class AddTethering(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        k=15,
        particles=[],
        positions='current',
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.tether_particles(
                sim_object=sim, 
                particles=self_conf.particles, 
                k=self_conf.k, 
                positions=self_conf.positions,
            )
        )


class AddGlobalVariableDynamics(SimulationAction):
    stage = 'loop'
    _default_params = AttrDict(
        variable_name = None,
        final_value = None,
        inital_block = 0,
        final_block = None
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if inital_block <= sim.block <= self_conf.final_block:
            cur_val = sim.context.getParameter(self_conf.variable_name)

            new_val = cur_val + (
                    (self_conf.final_value - cur_val) 
                    / (final_block - sim.block + 1)
                    )
            sim.context.setParameter(self_conf.variable_name, new_val)


class SaveConformation(SimulationAction):
    stage = 'loop'

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]
        path = os.path.join(shared_config['folder'], f'block.{sim.block}.txt.gz')
        data = sim.get_data()
        np.savetxt(path, data)

        return sim


#class AddPerParticleVariableDynamics(SimulationAction):
#    stage = 'init'
