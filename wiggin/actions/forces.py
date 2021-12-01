import numbers

from typing import Optional, Any, Union
# import dataclasses
from dataclasses import dataclass

import numpy as np

from ..core import SimAction, ConfigEntry
from .. import extra_forces

import polychrom


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

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=self.asdict())

        if hasattr(newconf.action["chains"], "__iter__") and hasattr(
            newconf.action["chains"][0], "__iter__"
        ):
            newconf.shared["chains"] = newconf.action["chains"]
        elif hasattr(newconf.action["chains"], "__iter__") and isinstance(
            newconf.action["chains"][0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(newconf.action["chains"])]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            newconf.action["chains"] = chains
            newconf.shared["chains"] = chains

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared

        nonbonded_force_func = None
        nonbonded_force_kwargs = {}
        if config.action["repulsion_e"]:
            if config.action["attraction_e"] and config.action["attraction_r"]:
                nonbonded_force_func = extra_forces.quartic_repulsive_attractive
                nonbonded_force_kwargs = dict(
                    repulsionEnergy=config.action["repulsion_e"],
                    repulsionRadius=1.0,
                    attractionEnergy=config.action["attraction_e"],
                    attractionRadius=config.action["attraction_r"],
                )

            else:
                nonbonded_force_func = extra_forces.quartic_repulsive
                nonbonded_force_kwargs = {"trunc": config.action["repulsion_e"]}

        sim.add_force(
            polychrom.forcekits.polymer_chains(
                sim,
                chains=config.shared["chains"],
                bond_force_func=polychrom.forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": config.action["bond_length"],
                    "bondWiggleDistance": config.action["wiggle_dist"],
                },
                angle_force_func=(
                    None if config.action["stiffness_k"] is None else polychrom.forces.angle_force
                ),
                angle_force_kwargs={"k": config.action["stiffness_k"]},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=config.action["except_bonds"],
            )
        )


@dataclass
class AddCylindricalConfinement(SimAction):
    k: float = 0.5
    r: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.add_force(
            polychrom.forces.cylindrical_confinement(
                sim_object=sim,
                r=config.action["r"],
                top=config.action["top"],
                bottom=config.action["bottom"],
                k=config.action["k"],
            )
        )


@dataclass
class AddSphericalConfinement(SimAction):
    k: float = 5
    r: Optional[Union[str, float]] = "density"
    density: Optional[float] = 1.0 / ((1.5) ** 3)

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared

        sim.add_force(
            polychrom.forces.spherical_confinement(
                sim,
                r=config.action["r"],  # radius... by default uses certain density
                k=config.action["k"],  # How steep the walls are
                density=config.action["density"],  # target density, measured in particles
                # per cubic nanometer (bond size is 1 nm)
                # name='spherical_confinement'
            )
        )


@dataclass
class AddTethering(SimAction):
    k: float = 15
    particles: Any = []
    positions: Any = "current"

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.add_force(
            polychrom.forces.tether_particles(
                sim_object=sim,
                particles=config.action["particles"],
                k=config.action["k"],
                positions=config.action["positions"],
            )
        )

