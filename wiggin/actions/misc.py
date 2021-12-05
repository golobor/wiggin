from dataclasses import dataclass

from typing import Any
from ..core import SimAction

import polychrom
import polychrom.forces


@dataclass
class CrosslinkParallelChains(SimAction):
    chains: Any = None
    bond_length: float = 1.0
    wiggle_dist: float = 0.025

    _shared = dict(N=None)

    def configure(self):
        if self.chains is None:
            self.chains = [
                (
                    (0, self._shared["N"] // 2, 1),
                    (
                        self._shared["N"] // 2,
                        self._shared["N"],
                        1,
                    ),
                ),
            ]

        return {}

    def run_init(self, sim):
        # do not use self.args!
        # only use parameters from config.action and config.shared

        bonds = sum(
            [
                zip(
                    range(chain1[0], chain1[1], chain1[2]),
                    range(chain2[0], chain2[1], chain2[2]),
                )
                for chain1, chain2 in self.chains
            ]
        )

        sim.add_force(
            polychrom.forces.harmonic_bonds(
                sim,
                bonds=bonds,
                bondLength=self.bond_length,
                bondWiggleDistance=self.wiggle_dist,
                name="ParallelChainsCrosslinkBonds",
            )
        )
