import copy
import os
import logging
import inspect
import typing

from dataclasses import dataclass, field

# import numpy as np

import polychrom
import polychrom.simulation

_VERBOSE = True
logging.basicConfig(level=logging.INFO)


class ConfigEntry(typing.NamedTuple):
    shared: dict
    action: dict


@dataclass
class SimAction:
    name: int = field(init=False)

    def __post_init__(self):
        self.name = type(self).__name__

    def set_name(self, new_name):
        self.name = new_name
        return self

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))
        return newconf

    def spawn_actions(self):
        pass

    # def __init__(self):
    # This line must be the first in the function:
    #     super().__init__(**locals())

    # def run_init(self, config:ConfigEntry, sim):
    #     # do not use self.params!
    #     # only use parameters from config.action and config.shared

    # def run_loop(self, config:ConfigEntry, sim):
    #     # do not use self.params!
    #     # only use parameters from config.action and config.shared


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

        self.action_params = dict()
        self._default_action_params = dict()

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
        if action.name in self.action_params:
            raise ValueError(
                f"Action {action.name} was already added to the constructor!"
            )

        self.action_params[action.name] = copy.deepcopy(action.params)

        self._default_action_params[action.name] = {
            k: v.default
            for k, v in inspect.signature(action.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        if len(order) != 3:
            raise ValueError("order must be a tuple of three numbers or Nones")

        order = tuple([len(self._actions) if i is None else i for i in order])

        self._actions.append((order, action))

    def configure(self) -> ConfigEntry:
        # sorted uses a stable sorting algorithm
        for order, action in sorted(self._actions, key=lambda x: x[0][0]):
            if _VERBOSE:
                logging.info(f"Configuring action {action.name}...")

            if action.name in self.newconf.actions:
                raise ValueError(f"Action {action.name} has already been configured!")

            newconf = action.configure(self.config.shared, self.newconf.actions)

            assert (
                newconf is not None
            ), f"{action.name}.configure() a ConfigEntry !"
            self.config['shared'].update(newconf.shared)
            self.config['actions'][action.name] = newconf.action

    def _run_action(self, action: SimAction, stage: str = 'init'):
        run_f_name = f'run_{stage}'
        if not hasattr(action, run_f_name):
            return True
        
        out = getattr(action, run_f_name)(
            ConfigEntry(shared=self.config['shared'],
                        action=self.newconf['actions'][action.name]),
            self._sim
        )

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

    def run(self):
        for order, action in sorted(self._actions, key=lambda x: x[0][1]):
            self._run_action(action, stage='init')
                
        while True:
            for order, action in sorted(self._actions, key=lambda x: x[0][2]):
                self._run_action(action, stage='init')

    def auto_name(self, root_data_folder="./data/"):
        name = []
        for action_name, params in self.action_params.items():
            default_params = self._default_action_params.get(action_name, {})
            for k, v in params.items():
                if k in default_params and v != default_params[k]:
                    name += ["_", k, "-", str(v)]

        name = "".join(name[1:])
        self.config.shared["name"] = name
        self.config.shared["folder"] = os.path.join(root_data_folder, name)


