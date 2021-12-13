import copy
import dataclasses
import os
import logging
import shelve
# import typing

from dataclasses import dataclass, field

# import numpy as np

import polychrom
import polychrom.simulation

_VERBOSE = True
logging.basicConfig(level=logging.INFO)


def _get_dataclass_defaults(dc):
    if not dataclasses.is_dataclass(dc):
        raise ValueError('`dc` must be a dataclass!')
    elif not isinstance(dc.__class__, type):
        dc = dc.__class__
    out = {}
    for k,v in dc.__dataclass_fields__.items():
        if not isinstance(v.default, dataclasses._MISSING_TYPE):
            out[k] = v.default
        elif not isinstance(v.default_factory, dataclasses._MISSING_TYPE):
            out[k] = v.default_factory()
    return out


@dataclass
class SimAction:
    name: int = field(init=False)

    _reads_shared = []
    _writes_shared = []
    _shared = dict()

    def __post_init__(self):
        self.name = type(self).__name__

    def store_shared(self, shared_config):
        for k in self._reads_shared:
            self._shared[k] = shared_config[k]

    def rename(self, new_name):
        self.name = new_name
        return self

    def configure(self):
        out_shared = {}
        return out_shared

    def spawn_actions(self):
        pass
    
    # def run_init(self, sim):
    #     pass

    # def run_loop(self, sim):
    #     pass

class SimConstructor:
    def __init__(self, name=None, folder=None):
        self._actions = []

        self._sim = None

        self.config = {}
        self.config['shared'] = dict(
            name=name,
            N=None,
            initial_conformation=None,
            folder=folder,
        )
        self.config['actions'] = dict()

        self.action_args = dict()
        self._default_action_args = dict()

    def add_action(self, action: SimAction, order=(None, None, None)):
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
        if action.name in self.action_args:
            raise ValueError(
                f"Action {action.name} was already added to the constructor!"
            )

        self.action_args[action.name] = copy.deepcopy(action.__dict__)

        self._default_action_args[action.name] = _get_dataclass_defaults(
            action.__class__)

        if len(order) != 3:
            raise ValueError("order must be a tuple of three numbers or Nones")

        order = tuple([len(self._actions) if i is None else i for i in order])

        self._actions.append((order, action))


    def configure(self):
        # sorted uses a stable sorting algorithm
        for order, action in sorted(self._actions, key=lambda x: x[0][0]):
            if _VERBOSE:
                logging.info(f"Configuring action {action.name}...")

            if action.name in self.config['actions']:
                raise ValueError(f"Action {action.name} has already been configured!")

            # populate the dictionary of shared parameters of the action.
            action.store_shared(self.config['shared'])

            out_shared = action.configure()

            if isinstance(out_shared, dict):
                promised_keys = set(action._writes_shared)
                provided_keys = set(out_shared.keys())
                if promised_keys != provided_keys:
                    raise RuntimeError(
                        f'The action {type(action)} is expected to set shared keys '
                        f'{promised_keys}, but instead provides {provided_keys}')
                self.config['shared'].update(out_shared)
                action._shared.update(out_shared)
            
            self.config['actions'][action.name] = {
                k:copy.deepcopy(v) 
                for k,v in action.__dict__.items()
                if k not in ['_shared', 'name']
            }
            

    def _run_action(self, action: SimAction, stage: str = 'init'):
        run_f_name = f'run_{stage}'
        if not hasattr(action, run_f_name):
            return True

        out = getattr(action, run_f_name)(self._sim)

        if issubclass(type(out), polychrom.simulation.Simulation):
            self._sim = out
            return True
        elif out is None:
            return True
        elif out is False:
            return False
        else:
            raise ValueError(
                f"{action.name}.run_{stage}() returned {out}. "
                "Allowed values are: polychrom.simulation.Simulation, None or False"
            )
        

    def run_init(self):
        for order, action in sorted(self._actions, key=lambda x: x[0][1]):
            self._run_action(action, stage='init')


    def run_loop(self):
        while True:
            for order, action in sorted(self._actions, key=lambda x: x[0][2]):
                self._run_action(action, stage='loop')


    def run(self):
        self.run_init()
        self.run_loop()
        

    def auto_name_folder(self, root_data_folder="./data/"):
        name = []
        for action_name, args in self.action_args.items():
            default_args = self._default_action_args.get(action_name, {})
            for k, v in args.items():
                if k in default_args and v != default_args[k]:
                    name += ["_", k, "-", str(v)]

        name = "".join(name[1:])
        self.config['shared']['name'] = name
        self.config['shared']['folder'] = os.path.join(root_data_folder, name)


    def save_config(self, backup=True, mode_exists='fail'):
        folder = self.config['shared']['folder']
        if mode_exists not in ["fail", "overwrite"]:
            raise ValueError(
                f'Unknown mode for saving configuration: {mode_exists}'
            )

        if os.path.exists(folder):
            if (mode_exists == "fail"):
                raise OSError(
                    f'The output folder already exists {folder}'
                )
            else:
                logging.info(f'removing previously existing config files in {folder}')

        os.makedirs(folder, exist_ok=True)
        paths = [os.path.join(folder, "conf.shlv")]

        if backup:
            os.mkdir(os.path.join(folder, "backup"))
            paths.append(os.path.join(folder, "backup", "conf.shlv"))

        for path in paths:
            with shelve.open(path, protocol=2) as conf:
                conf.clear()
                conf['config'] = self.config
                conf['action_args'] = self.action_args