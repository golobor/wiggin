from dataclasses import dataclass

from ..core import SimAction, ConfigEntry

import polychrom
import polychrom.starting_conformations


@dataclass
class GenerateRWInitialConformation(SimAction):
    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=self.asdict())

        newconf.shared[
            "initial_conformation"
        ] = polychrom.starting_conformations.create_random_walk(
            step_size=1.0, N=config.shared["N"]
        )

        return newconf
