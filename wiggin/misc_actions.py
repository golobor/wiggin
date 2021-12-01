import copy
from .core import SimAction, ConfigEntry

import polychrom
import polychrom.forces


class CrosslinkParallelChains(SimAction):
    def __init__(
        self,
        chains=None,
        bond_length=1.0,
        wiggle_dist=0.025,
    ):
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        if newconf.action["chains"] is None:
            newconf.action["chains"] = [
                (
                    (0, newconf.shared["N"] // 2, 1),
                    (
                        newconf.shared["N"] // 2,
                        newconf.shared["N"],
                        1,
                    ),
                ),
            ]

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared

        bonds = sum(
            [
                zip(
                    range(chain1[0], chain1[1], chain1[2]),
                    range(chain2[0], chain2[1], chain2[2]),
                )
                for chain1, chain2 in config.action["chains"]
            ]
        )

        sim.add_force(
            polychrom.forces.harmonic_bonds(
                sim,
                bonds=bonds,
                bondLength=config.action["bond_length"],
                bondWiggleDistance=config.action["wiggle_dist"],
                name="ParallelChainsCrosslinkBonds",
            )
        )
