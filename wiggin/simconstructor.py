import socket
import copy
import os
import logging
import inspect
import numbers

import bisect

import numpy as np

# import numpy as np

from polychrom import simulation, forces, forcekits, hdf5_format, starting_conformations
from . import extra_forces


_VERBOSE = True
logging.basicConfig(level=logging.INFO)


class SimConstructor:
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
        self._default_action_params = dict()
        self.action_configs = dict()


    def add_action(self, action, order=(None, None, None)):
        """
        Add an action to the constructor.
        
        Parameters:
            action: SimAction
            order: (float, float, float)
                If provided, the three numbers specify the order of the execution of 
                .configure(), .run_init() and .run_loop(). If not provided,
                the order of execution is calculated based on the order of addition
                of actions into the system.
                Use at your peril!

        """
        if action.name in self.action_params:
            raise ValueError(
                f'Action {action.name} was already added to the constructor!')
        self.action_params[action.name] = copy.deepcopy(action.params)
        self._default_action_params[action.name] = {
                k: v.default
                for k, v in inspect.signature(action.__init__).parameters.items()
                if v.default is not inspect.Parameter.empty
        }

        if len(order) != 3:
            raise ValueError('order must be a tuple of three numbers or Nones')
        order = tuple([len(self._actions) if i is None else i for i in order])
        
        self._actions.append((order,action))


    def configure(self):
        for order, action in sorted(self._actions, key=lambda x: x[0][0]):
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
        for order, action in sorted(self._actions, key=lambda x: x[0][1]):
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
            for order, action in sorted(self._actions, key=lambda x: x[0][2]):
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
                        return
                    else:
                        raise ValueError(f'{action.name}.run_loop() returned {new_sim}. '
                                        'Allowed values are: polychrom.simulation.Simulation, None or False')



    def auto_name(self, root_data_folder = './data/'):
        name = []
        for action_name, params in self.action_params.items():
            default_params = self._default_action_params.get(action_name, {})
            for k, v in params.items():
                if k in default_params and v != default_params[k]:
                    name += ['_', k, '-', str(v)]

        name = ''.join(name[1:])
        self.shared_config['name'] = name
        self.shared_config['folder'] = os.path.join(root_data_folder, name)
        

class SimAction:
    def __init__(
            self, 
            **kwargs
            ):

        self.name = type(self).__name__
        kwargs.pop('self', None)
        kwargs.pop('__class__', None)
        self.params = dict(kwargs)


    def set_name(self, new_name):
        self.name = new_name
        return self


    def configure(self, shared_config, action_configs):
        shared_config_added_data = dict()
        action_config = copy.deepcopy(self.params)

        return shared_config_added_data, action_config
        

    # def __init__(self):
    #     params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
    #     super().__init__(**params)

    # def run_init(self, shared_config, action_configs, sim):
    #     # do not use self.params!
    #     # only use parameters from action_configs[self.name] and shared_config
    #     self_conf = action_configs[self.name]



    # def run_loop(self, shared_config, action_configs, sim):
    #     # do not use self.params!
    #     # only use parameters from action_configs[self.name] and shared_config
    #     self_conf = action_configs[self.name]



class InitializeSimulation(SimAction):
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
            reporter_block_size=10,
            reporter_blocks_only=False,
            ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.

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
                'SimConstructor() or via NameSimulationByParams()'
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
            timestep=self_conf['timestep'],
            collision_rate=self_conf['collision_rate'],
            mass=self_conf['mass'],
            PBCbox=self_conf['PBCbox'],
            N=shared_config['N'],
            max_Ek=self_conf['max_Ek'],
            reporters = [reporter]
        )  

        return sim


class BlockStep(SimAction):
    def __init__(
        self,
        num_blocks = 100,
        block_size = int(1e4)
        ):

        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if (sim.step / self_conf['block_size'] < self_conf['num_blocks']):
            sim.do_block(self_conf['block_size'])
            return sim
        else:
            return False




class LocalEnergyMinimization(SimAction):
    def __init__(
        self,
        max_iterations = 1000,
        tolerance = 1,
        random_offset = 0.1
        ):

        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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
 

class AddChains(SimAction):
    def __init__(
        self,
        chains = ((0, None, 0),),
        bond_length = 1.0,
        wiggle_dist = 0.025,
        stiffness_k = None,
        repulsion_e = 2.5, ## TODO: implement np.inf 
        attraction_e = None,
        attraction_r = None,
        except_bonds = False,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if hasattr(action_config['chains'], '__iter__') and hasattr(action_config['chains'][0], '__iter__'):
            shared_config_added_data['chains'] = action_config['chains']
        elif hasattr(action_config['chains'], '__iter__') and isinstance(action_config['chains'][0], numbers.Number):
            edges = np.r_[0, np.cumsum(action_config['chains'])]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            action_config['chains'] = chains
            shared_config_added_data['chains'] = chains
            
        return shared_config_added_data, action_config


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        nonbonded_force_func = None
        nonbonded_force_kwargs = {}
        if self_conf['repulsion_e']:
            if self_conf['attraction_e'] and self_conf['attraction_r']:
                nonbonded_force_func = extra_forces.quartic_repulsive_attractive
                nonbonded_force_kwargs = dict(
                    repulsionEnergy=self_conf['repulsion_e'],
                    repulsionRadius=1.0,
                    attractionEnergy=self_conf['attraction_e'],
                    attractionRadius=self_conf['attraction_r'],
                )

            else:
                nonbonded_force_func = extra_forces.quartic_repulsive
                nonbonded_force_kwargs = {
                    'trunc': self_conf['repulsion_e'] 
                }

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=shared_config['chains'],
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

                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,

                except_bonds=self_conf['except_bonds']
            )
        )


class CrosslinkParallelChains(SimAction):
    def __init__(
        self,
        chains = None,
        bond_length = 1.0,
        wiggle_dist = 0.025,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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


class GenerateRWInitialConformation(SimAction):
    def __init__(
        self
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        shared_config_added_data['initial_conformation'] = (
            starting_conformations.create_random_walk(step_size=1.0, N=shared_config['N'])
        )

        return shared_config_added_data, action_config


class SetInitialConformation(SimAction):

    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        # self_conf = action_configs[self.name]
        sim.set_data(shared_config['initial_conformation'])

        return sim


class AddCylindricalConfinement(SimAction):
    def __init__(
        self,
        k=0.5,
        r=None,
        top=None,
        bottom=None,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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



class AddSphericalConfinement(SimAction):
    def __init__(
        self,
        k=5,
        r='density',
        density= 1. / ((1.5)**3),
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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


class AddTethering(SimAction):
    def __init__(
        self,
        k=15,
        particles=[],
        positions='current',
        ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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


class AddGlobalVariableDynamics(SimAction):
    def __init__(
        self,
        variable_name = None,
        final_value = None,
        inital_block = 0,
        final_block = None
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
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

            logging.info(f'set {self_conf["variable_name"]} to {new_val}')
            sim.context.setParameter(self_conf['variable_name'], new_val)



class AddDynamicParameterUpdate(SimAction):
    def __init__(
        self,
        force,
        param,
        ts = [90, 100],
        vals = [0, 1.0], 
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)
        return shared_config_added_data, action_config


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        t = sim.block
        ts = self_conf['ts']
        vals = self_conf['vals']

        if ts[0] <= t <= ts[-1]:
            
            step = bisect.bisect_left(ts, t) - 1
            if step == -1:
                step = 0
            
            
            param_full_name = f'{self_conf["force"]}_{self_conf["param"]}'
            cur_val = sim.context.getParameter(param_full_name) 
            new_val = np.interp(t, ts[step:step+2], vals[step:step+2])
            
            if cur_val != new_val:
                 
                logging.info(f'set {param_full_name} to {new_val}')
                sim.context.setParameter(param_full_name, new_val)



# class SaveConformationTxt(SimAction):

#     def run_loop(self, shared_config, action_configs, sim):
#         # do not use self.params!
#         # only use parameters from action_configs[self.name] and shared_config
#         self_conf = action_configs[self.name]
#         path = os.path.join(shared_config['folder'], f'block.{sim.block}.txt.gz')
#         data = sim.get_data()
#         np.savetxt(path, data)

#         return sim
