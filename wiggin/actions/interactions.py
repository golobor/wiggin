from dataclasses import dataclass
import numbers

import numpy as np

from typing import Optional, Any, Union

import polychrom

from ..core import SimAction

from .. import forces


@dataclass
class AddChains(SimAction):
    chains: Any = ((0, None, 0),)
    bond_length: float = 1.0
    wiggle_dist: float = 0.025
    stiffness_k: Optional[float] = None
    repulsion_e: Optional[float] = 2.5
    attraction_e: Optional[float] = None
    attraction_r: Optional[float] = None
    except_bonds: Union[bool, int] = False

    def configure(self):
        out_shared = {}

        if hasattr(self.chains, "__iter__") and hasattr(
            self.chains[0], "__iter__"
        ):
            out_shared["chains"] = self.chains
        elif hasattr(self.chains, "__iter__") and isinstance(
            self.chains[0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(self.chains)]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            self.chains = chains
            out_shared['chains'] = chains

        return out_shared

    def run_init(self, sim):
        # do not use self.args!
        # only use parameters from self.ndonfig.shared

        nonbonded_force_func = None
        nonbonded_force_kwargs = {}
        if self.repulsion_e:
            if self.attraction_e and self.attraction_r:
                nonbonded_force_func = forces.quartic_repulsive_attractive
                nonbonded_force_kwargs = dict(
                    repulsionEnergy=self.repulsion_e,
                    repulsionRadius=1.0,
                    attractionEnergy=self.attraction_e,
                    attractionRadius=self.attraction_r,
                )

            else:
                nonbonded_force_func = forces.quartic_repulsive
                nonbonded_force_kwargs = {"trunc": self.repulsion_e}

        sim.add_force(
            polychrom.forcekits.polymer_chains(
                sim,
                chains=self.chains,
                bond_force_func=polychrom.forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": self.bond_length,
                    "bondWiggleDistance": self.wiggle_dist,
                },
                angle_force_func=(
                    None if self.stiffness_k is None else polychrom.forces.angle_force
                ),
                angle_force_kwargs={"k": self.stiffness_k},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=self.except_bonds,
            )
        )
