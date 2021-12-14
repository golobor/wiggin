import os
import socket
import bisect
import logging

from typing import Optional, Tuple, Sequence, Union
from dataclasses import dataclass

from ..core import SimAction

import polychrom
import polychrom.simulation
import polychrom.forces
import polychrom.forcekits
import polychrom.hdf5_format


@dataclass
class InitializeSimulation(SimAction):
    N: Optional[int] = None
    computer_name: Optional[str] = None
    platform: str = "CUDA"
    GPU: Optional[str] = "0"
    integrator: str = "variableLangevin"
    error_tol: float = 0.01
    mass: float = 1
    collision_rate: float = 0.003
    temperature: float = 300
    timestep: float = 1.0
    max_Ek: float = 1000
    PBCbox: Union[bool, Tuple[float, float, float]] = False
    reporter_block_size: int = 10
    reporter_blocks_only: bool = False

    _reads_shared = ['folder']
    _writes_shared = ['N']

    def configure(self):
        out_shared = {}

        self.computer_name = socket.gethostname()
        if self.N is not None:
            out_shared['N'] = self.N

        return out_shared

    def run_init(self, sim):
        if self._shared['folder'] is None:
            raise ValueError(
                "The data folder is not set"
            )

        os.makedirs(self._shared['folder'], exist_ok=True)

        reporter = polychrom.hdf5_format.HDF5Reporter(
            folder=self._shared['folder'],
            max_data_length=self.reporter_block_size,
            blocks_only=self.reporter_blocks_only,
            overwrite=False,
        )

        sim = polychrom.simulation.Simulation(
            platform=self.platform,
            GPU=self.GPU,
            integrator=self.integrator,
            error_tol=self.error_tol,
            timestep=self.timestep,
            collision_rate=self.collision_rate,
            mass=self.mass,
            PBCbox=self.PBCbox,
            N=self.N,
            max_Ek=self.max_Ek,
            reporters=[reporter],
        )

        return sim


@dataclass
class SetInitialConformation(SimAction):
    _reads_shared = ['initial_conformation']

    def run_init(self, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.set_data(self._shared["initial_conformation"])

        return sim


@dataclass
class BlockStep(SimAction):
    num_blocks: int = 100
    block_size: int = int(1e4)

    def run_loop(self, sim):
        if sim.step / self.block_size < self.num_blocks:
            sim.do_block(self.block_size)
            return sim
        else:
            return False


@dataclass
class LocalEnergyMinimization(SimAction):
    max_iterations: int = 1000
    tolerance: float = 1
    random_offset: float = 0.1

    def run_init(self, sim):
        sim.local_energy_minimization(
            maxIterations=self.max_iterations,
            tolerance=self.tolerance,
            random_offset=self.random_offset,
        )


def _interpolate(t, ts, vals, power=1):
    if (t < ts[0]) or (t > ts[-1]):
        return None

    step = max(0, bisect.bisect_left(ts, t) - 1)

    t0, t1 = ts[step:step+2]

    v0, v1 = vals[step:step+2]
    if power == 1:
        new_val = v0 + (v1 - v0) * ((t-t0) / (t1-t0))
    else:
        v0, v1 = v0 ** power, v1 ** power
        new_val = v0 + (v1 - v0) * ((t-t0) / (t1-t0))
        new_val = new_val ** (1/power)

    return new_val


@dataclass
class UpdateGlobalParameter(SimAction):
    force: str = ''
    param: str = ''
    ts: Sequence[float] = (90, 100)
    vals: Sequence[float] = (0, 1.0)
    power: float = 1.0

    def run_loop(self, sim):
        new_val = _interpolate(t=sim.block, ts=self.ts, vals=self.vals, power=self.power)

        if new_val is None:
            return

        if self.force:
            param_full_name = f'{self.force}_{self.param}'
        else:
            param_full_name = self.param

        cur_val = sim.context.getParameter(param_full_name)
        
        if cur_val != new_val:
            logging.info(f"set {param_full_name} to {new_val}")
            sim.context.setParameter(param_full_name, new_val)


@dataclass
class UpdatePerParticleParameter(SimAction):
    force: str = ''
    parameter_name: str = ''
    term_index: Optional[int] = None
    particle_index: Optional[int] = None
    ts: Sequence[float] = (90, 100)
    vals: Sequence[float] = (0, 1.0)
    power: float = 1.0

    def configure(self):
        if (self.particle_index is None) == (self.term_index is None):
            raise ValueError('Provide either a particle index or a term index')
        return {}

    def run_loop(self, sim):
        new_val = _interpolate(t=sim.block, ts=self.ts, vals=self.vals, power=self.power)

        if new_val is None:
            return

        force_obj = sim.force_dict[self.force]

        param_index = [
            force_obj.getPerParticleParameterName(i)
            for i in range(force_obj.getNumPerParticleParameters())
            ].index(self.parameter_name)
        
        if self.term_index is None:
            params = [
                force_obj.getParticleParameters(self, i)
                for i in range(force_obj.getNumParticles())
            ]
            particle_index = self.particle_index
            term_index, particle_params = [
                i for i in params if i[0] == particle_index]
        else:
            term_index = self.term_index
            particle_index, particle_params = force_obj.getParticleParameters(term_index)

        cur_val = particle_params[param_index]

        if cur_val == new_val:
            return

        particle_params = list(particle_params)
        particle_params[param_index] = new_val
        particle_params = tuple(particle_params)

        force_obj.setParticleParameters(
            term_index, particle_index, particle_params)
        force_obj.updateParametersInContext(sim.context)

        logging.info(f"set {self.parameter_name} of force {self.force} to {new_val}")



# move to methods of the SimulationConstructor


# DEPRECATED in favor of AddDynamicParameterUpdate
# class AddGlobalVariableDynamics(SimAction):
#     def __init__(
#         self,
#         variable_name=None,
#         final_value=None,
#         inital_block=0,
#         final_block=None
#     ):
#         params = {
#             k: v for k, v in locals().items() if k not in ["self"]
#         }  # This line must be the first in the function.
#         super().__init__(**locals())

#     def run_loop(self, config:ConfigEntry, sim):
#         # do not use self.params!
#         # only use parameters from config.action and config.shared

#         if config.action["inital_block"] <= sim.block <= config.action["final_block"]:
#             cur_val = sim.context.getParameter(config.action["variable_name"])

#             new_val = cur_val + (
#                 (config.action["final_value"] - cur_val)
#                 / (config.action["final_block"] - sim.block + 1)
#             )

#             logging.info(f'set {config.action["variable_name"]} to {new_val}')
#             sim.context.setParameter(config.action["variable_name"], new_val)
