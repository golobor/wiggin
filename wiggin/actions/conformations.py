from dataclasses import dataclass

from ..core import SimAction

import polychrom
import polychrom.starting_conformations


@dataclass
class RandomWalkConformation(SimAction):
    _reads_shared = ['N']
    _writes_shared = ['initial_conformation']

    def configure(self):
        out_shared = {}

        out_shared[
            "initial_conformation"
        ] = polychrom.starting_conformations.create_random_walk(
            step_size=1.0, N=self._shared["N"]
        )

        return out_shared

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.set_data(self._shared["initial_conformation"])

        return sim
