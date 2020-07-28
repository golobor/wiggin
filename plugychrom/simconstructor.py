import socket
import copy
import os
import logging

# import numpy as np

from polychrom import simulation, forces, forcekits, hdf5_format
from . import extra_forces


_VERBOSE = True
logging.basicConfig(level=logging.INFO)


class SimulationConstructor:
    def __init__(self, name=None, folder=None):
        self._actions = []

        self._sim = None
        self.shared_config = dict(
            name=name,
            N=None,
            initial_conformation=None,
            folder=folder,
        )

        self.action_params = dict()
        self.action_configs = dict()


    def add_action(self, action):
        if action.name in self.action_params:
            raise ValueError(
                f'Action {action.name} was already added to the constructor!')
        self.action_params[action.name] = copy.deepcopy(action.params)

        self._actions.append(action)


    def configure(self):
        for action in self._actions:
            if _VERBOSE:
                logging.info(f'Configuring action {action.name}...')

            if action.name in self.action_configs:
                raise ValueError(
                    f'Action {action.name} has already been configured!')

            conf_res = action.configure(
                self.shared_config,
                self.action_configs)

            assert conf_res is not None, (
                    f'{action.name}.configure() must return two dict() configs!')

            shared_config_added_data, action_config = conf_res
            self.shared_config.update(shared_config_added_data)
            self.action_configs[action.name] = action_config


    def run(self):
        for action in self._actions:
            if hasattr(action, 'run_init'):
                new_sim = action.run_init(
                        self.shared_config, 
                        self.action_configs, 
                        self._sim)
                if new_sim is None:
                    continue
                elif issubclass(type(new_sim), simulation.Simulation):
                    self._sim = new_sim
                elif new_sim is False:
                    return
                else:
                    raise ValueError(f'{action.name}.run_init() returned {new_sim}. '
                                     'Allowed values are: polychrom.simulation.Simulation, None or False')



        while True:
            for action in self._actions:
                if hasattr(action, 'run_loop'):
                    new_sim = action.run_loop(
                            self.shared_config, 
                            self.action_configs, 
                            self._sim)

                    if new_sim is None:
                        continue
                    elif issubclass(type(new_sim), simulation.Simulation):
                        self._sim = new_sim
                    elif new_sim is False:
                        break
                    else:
                        raise ValueError(f'{action.name}.run_loop() returned {new_sim}. '
                                        'Allowed values are: polychrom.simulation.Simulation, None or False')



    def auto_name(self, root_data_folder = './data/'):
        name = []
        for _, params in self.action_params.items():
            for k,v in params.items():
                name += ['_', k, '-', str(v)]

        name = ''.join(name[1:])
        self.shared_config['name'] = name
        self.shared_config['folder'] = os.path.join(root_data_folder, name)
        

class SimulationAction:
    def __init__(
            self, 
            **kwargs
            ):

        self.name = type(self).__name__
        kwargs.pop('self', None)
        self.params = dict(kwargs)


    def set_name(self, new_name):
        self.name = new_name
        return self


    def configure(self, shared_config, action_configs):
        shared_config_added_data = dict()
        action_config = copy.deepcopy(self.params)

        return shared_config_added_data, action_config
        

    # def __init__(self):
    #     params = dict(locals()) # Must be the very first line of the function!
    #     super().__init__(**params)

    # def run_init(self, shared_config, action_configs, sim):
    #     # do not use self.params!
    #     # only use parameters from action_configs[self.name] and shared_config
    #     self_conf = action_configs[self.name]



    # def run_loop(self, shared_config, action_configs, sim):
    #     # do not use self.params!
    #     # only use parameters from action_configs[self.name] and shared_config
    #     self_conf = action_configs[self.name]



class InitializeSimulation(SimulationAction):
    def __init__(
            self,
            N=None,
            computer_name=None,
            platform='CUDA',
            GPU='0',
            integrator='variableLangevin',
            error_tol=0.01,
            mass=1,
            collision_rate=0.003,
            temperature=300,
            timestep=1.0,
            max_Ek=1000,
            PBCbox=False,
            reporter_block_size=50,
            reporter_blocks_only=False,
            ):
        params = dict(locals()) # Must be the very first line of the function!

        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        action_config['computer_name'] = socket.gethostname()
        if action_config['N'] is not None:
            shared_config_added_data['N'] = action_config['N']

        return shared_config_added_data, action_config


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config

        self_conf = action_configs[self.name]

        if shared_config['folder'] is None:
            raise ValueError(
                'The data folder is not set, please specify it in '
                'SimulationConstructor() or via NameSimulationByParams()'
            )

        os.makedirs(shared_config['folder'], exist_ok=True)
        
        reporter = hdf5_format.HDF5Reporter(
            folder=shared_config['folder'],
            max_data_length=self_conf['reporter_block_size'], 
            blocks_only=self_conf['reporter_blocks_only'],
            overwrite=False)

        sim = simulation.Simulation(
            platform=self_conf['platform'],
            GPU=self_conf['GPU'],
            integrator=self_conf['integrator'],
            error_tol=self_conf['error_tol'],
            collision_rate=self_conf['collision_rate'],
            mass=self_conf['mass'],
            PBCbox=self_conf['PBCbox'],
            N=shared_config['N'],
            max_Ek=self_conf['max_Ek'],
            reporters = [reporter]
        )  

        return sim


class BlockStep(SimulationAction):
    
    def __init__(
        self,
        num_blocks = 100,
        block_size = int(1e4)
        ):

        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if (sim['step'] / self_conf['block_size'] < self_conf['num_blocks']):
            sim.do_block(self_conf['block_size'])
            return sim
        else:
            return False


class LocalEnergyMinimization(SimulationAction):
    def __init__(
        self,
        max_iterations = 1000,
        tolerance = 1,
        random_offset = 0.1
        ):

        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]
        sim.local_energy_minimization(
                maxIterations=self_conf['max_iterations'],
                tolerance=self_conf['tolerance'],
                random_offset=self_conf['random_offset']
        )
 

class AddChains(SimulationAction):
    def __init__(
        self,
        chains = ((0, None, 0)),
        bond_length = 1.0,
        wiggle_dist = 0.025,
        stiffness_k = None,
        repulsion_e = 2.5, ## TODO: implement np.inf 
        except_bonds = False,
    ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=self_conf['chains'],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength': self_conf['bond_length'],
                    'bondWiggleDistance': self_conf['wiggle_dist'],
                },

                angle_force_func=(
                    None if self_conf['stiffness_k'] is None 
                    else forces.angle_force),
                angle_force_kwargs={
                    'k': self_conf['stiffness_k'] 
                },

                nonbonded_force_func=(
                    None if self_conf['repulsion_e'] is None 
                    else extra_forces.quartic_repulsive),
                nonbonded_force_kwargs={
                    'trunc': self_conf['repulsion_e'] 
                },

                except_bonds=self_conf['except_bonds']
            )
        )


class CrosslinkParallelChains(SimulationAction):
    def __init__(
        self,
        chains = None,
        bond_length = 1.0,
        wiggle_dist = 0.025,
    ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


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


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        bonds = sum([
            zip(range(chain1[0], chain1[1], chain1[2]),
                range(chain2[0], chain2[1], chain2[2]))
            for chain1, chain2 in self_conf['chains']
        ])

        sim.add_force(
            forces.harmonic_bonds(
                sim,
                bonds=bonds,
                bondLength= self_conf['bond_length'],
                bondWiggleDistance= self_conf['wiggle_dist'],
                name='ParallelChainsCrosslinkBonds'
            )
        )


class SetInitialConformation(SimulationAction):

    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        # self_conf = action_configs[self.name]
        sim.set_data(shared_config['initial_conformation'])

        return sim


class AddCylindricalConfinement(SimulationAction):
    def __init__(
        self,
        k=0.5,
        r=None,
        top=None,
        bottom=None,
    ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.cylindrical_confinement(
                sim_object=sim,
                r=self_conf['r'],
                top=self_conf['top'],
                bottom=self_conf['bottom'], 
                k=self_conf['k']
            )
        )



class AddSphericalConfinement(SimulationAction):
    def __init__(
        self,
        k=5,
        r='density',
        density= 1. / ((1.5)**3),
    ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.spherical_confinement(
                sim,
                r=self_conf['r'],  # radius... by default uses certain density
                k=self_conf['k'],  # How steep the walls are
                density=self_conf['density'],    # target density, measured in particles
                                              # per cubic nanometer (bond size is 1 nm)
                # name='spherical_confinement'
            )
        )


class AddTethering(SimulationAction):
    def __init__(
        self,
        k=15,
        particles=[],
        positions='current',
        ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.tether_particles(
                sim_object=sim, 
                particles=self_conf['particles'], 
                k=self_conf['k'], 
                positions=self_conf['positions'],
            )
        )


class AddGlobalVariableDynamics(SimulationAction):
    def __init__(
        self,
        variable_name = None,
        final_value = None,
        inital_block = 0,
        final_block = None
    ):
        params = dict(locals()) # Must be the very first line of the function!
        super().__init__(**params)


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if self_conf['inital_block'] <= sim.block <= self_conf['final_block']:
            cur_val = sim.context.getParameter(self_conf['variable_name'])

            new_val = cur_val + (
                    (self_conf['final_value'] - cur_val) 
                    / (self_conf['final_block'] - sim.block + 1)
                    )
            sim.context.setParameter(self_conf['variable_name'], new_val)


# class SaveConformationTxt(SimulationAction):

#     def run_loop(self, shared_config, action_configs, sim):
#         # do not use self.params!
#         # only use parameters from action_configs[self.name] and shared_config
#         self_conf = action_configs[self.name]
#         path = os.path.join(shared_config['folder'], f'block.{sim.block}.txt.gz')
#         data = sim.get_data()
#         np.savetxt(path, data)

#         return sim
