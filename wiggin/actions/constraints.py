from typing import Optional, Any, Union
import dataclasses
from dataclasses import dataclass


from ..core import SimAction

import polychrom
import polychrom.forces
import polychrom.forcekits


@dataclass
class CylindricalConfinement(SimAction):
    k: float = 0.5
    r: Optional[float] = None
    top: Optional[float] = None
    bottom: Optional[float] = None

    def run_init(self, sim):
        # do not use self.args!
        # only use parameters from self.ndonfig.shared
        sim.add_force(
            polychrom.forces.cylindrical_confinement(
                sim_object=sim,
                r=self.r,
                top=self.top,
                bottom=self.bottom,
                k=self.k,
            )
        )


@dataclass
class SphericalConfinement(SimAction):
    k: float = 5
    r: Optional[Union[str, float]] = "density"
    density: Optional[float] = 1.0 / ((1.5) ** 3)

    def run_init(self, sim):
        # do not use self.args!
        # only use parameters from self.ndonfig.shared

        sim.add_force(
            polychrom.forces.spherical_confinement(
                sim,
                r=self.r,  # radius... by default uses certain density
                k=self.k,  # How steep the walls are
                density=self.density,  # target density, measured in particles
                # per cubic nanometer (bond size is 1 nm)
                # name='spherical_confinement'
            )
        )


@dataclass
class Tethering(SimAction):
    k: float = 15
    particles: Any = dataclasses.field(default_factory=lambda: [])
    positions: Any = "current"

    def run_init(self, sim):
        # do not use self.args!
        # only use parameters from self.ndonfig.shared
        sim.add_force(
            polychrom.forces.tether_particles(
                sim_object=sim,
                particles=self.particles,
                k=self.k,
                positions=self.positions,
            )
        )
