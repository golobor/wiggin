import copy
import os
import sys
import socket
import numbers
import shelve
import bisect
import logging

from typing import Optional, Tuple
from dataclasses import dataclass

from .core import SimAction, ConfigEntry
from . import extra_forces

import numpy as np

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
    PBCbox: Optional[Tuple[float, float, float]] = None
    reporter_block_size: int = 10
    reporter_blocks_only: bool = False

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=self.asdict())
        newconf.action["computer_name"] = socket.gethostname()
        if self.params["N"] is not None:
            newconf.shared["N"] = self.params["N"]

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        if config.shared["folder"] is None:
            raise ValueError(
                "The data folder is not set, please specify it in "
                "SimConstructor() or via NameSimulationByParams()"
            )

        os.makedirs(config.shared["folder"], exist_ok=True)

        reporter = polychrom.hdf5_format.HDF5Reporter(
            folder=config.shared["folder"],
            max_data_length=config.action["reporter_block_size"],
            blocks_only=config.action["reporter_blocks_only"],
            overwrite=False,
        )

        sim = polychrom.simulation.Simulation(
            platform=config.action["platform"],
            GPU=config.action["GPU"],
            integrator=config.action["integrator"],
            error_tol=config.action["error_tol"],
            timestep=config.action["timestep"],
            collision_rate=config.action["collision_rate"],
            mass=config.action["mass"],
            PBCbox=config.action["PBCbox"],
            N=config.shared["N"],
            max_Ek=config.action["max_Ek"],
            reporters=[reporter],
        )

        return sim


class BlockStep(SimAction):
    num_blocks: int = 100
    block_size: int = int(1e4)

    def run_loop(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared

        if sim.step / config.action["block_size"] < config.action["num_blocks"]:
            sim.do_block(config.action["block_size"])
            return sim
        else:
            return False


class LocalEnergyMinimization(SimAction):
    max_iterations: int = 1000
    tolerance: float = 1
    random_offset: float = 0.1

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
        sim.local_energy_minimization(
            maxIterations=config.action["max_iterations"],
            tolerance=config.action["tolerance"],
            random_offset=config.action["random_offset"],
        )


class AddChains(SimAction):
    def __init__(
        self,
        chains=((0, None, 0),),
        bond_length=1.0,
        wiggle_dist=0.025,
        stiffness_k=None,
        repulsion_e=2.5,  # TODO: implement np.inf
        attraction_e=None,
        attraction_r=None,
        except_bonds=False,
    ):
        
        super().__init__(**locals())

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


class GenerateRWInitialConformation(SimAction):
    def __init__(self):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=self.asdict())

        newconf.shared[
            "initial_conformation"
        ] = polychrom.starting_conformations.create_random_walk(
            step_size=1.0, N=config.shared["N"]
        )

        return newconf


class SetInitialConformation(SimAction):
    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared
    
        sim.set_data(config.shared["initial_conformation"])

        return sim


class AddCylindricalConfinement(SimAction):
    def __init__(
        self,
        k=0.5,
        r=None,
        top=None,
        bottom=None,
    ):
        
        super().__init__(**locals())

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


class AddSphericalConfinement(SimAction):
    def __init__(
        self,
        k=5,
        r="density",
        density=1.0 / ((1.5) ** 3),
    ):
        
        super().__init__(**locals())

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


class AddTethering(SimAction):
    def __init__(
        self,
        k=15,
        particles=[],
        positions="current",
    ):
        super().__init__(**locals())

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


class AddDynamicParameterUpdate(SimAction):
    def __init__(
        self,
        force,
        param,
        ts=[90, 100],
        vals=[0, 1.0],
        power=1.0,
    ):
        super().__init__(**locals())

    def run_loop(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from config.action and config.shared

        t = sim.block
        ts = config.action["ts"]

        if (ts[0] < t) or (t > ts[-1]):
            return

        vals = config.action["vals"]
        power = config.action['vals']

        if config.action["force"]:
            param_full_name = f'{config.action["force"]}_{config.action["param"]}'
        else:
            param_full_name = config.action["param"]

        cur_val = sim.context.getParameter(param_full_name)

        step = max(0, bisect.bisect_left(ts, t) - 1)

        t0, t1 = ts[step:step+2]
        v0, v1 = vals[step:step+2]

        new_val = v1 + (v0 - v1) * (((t0-t) / (t0-t1)) ** power)

        if cur_val != new_val:
            logging.info(f"set {param_full_name} to {new_val}")
            sim.context.setParameter(param_full_name, new_val)


class SaveConfiguration(SimAction):
    def __init__(
        self,
        backup=True,
        mode_exists="fail",  # 'exit' 'overwrite'
    ):
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=self.asdict())

        if self.params["mode_exists"] not in ["fail", "exit", "overwrite"]:
            raise ValueError(
                f'Unknown mode for saving configuration: {self.params["mode_exists"]}'
            )
        if os.path.exists(config.shared["folder"]):
            if self.params["mode_exists"] == "fail":
                raise OSError(
                    f'The output folder already exists {config.shared["folder"]}'
                )
            elif self.params["mode_exists"] == "exit":
                sys.exit(0)

        os.makedirs(config.shared["folder"], exist_ok=True)
        paths = [os.path.join(config.shared["folder"], "conf.shlv")]

        if self.params["backup"]:
            os.mkdir(config.shared["folder"] + "/backup/")
            paths.append(os.path.join(config.shared["folder"], "backup", "conf.shlv"))

        for path in paths:
            conf = shelve.open(path, protocol=2)
            # TODO: fix saving action_params
            # conf['params'] = {k:dict(v) for k,v in ..}
            conf['config'] = config
            conf.close()

        return newconf


# class SaveConformationTxt(SimAction):

#     def run_loop(self, config:ConfigEntry, sim):
#         # do not use self.params!
#         # only use parameters from config.action and config.shared
#      
#         path = os.path.join(config.shared['folder'], f'block.{sim.block}.txt.gz')
#         data = sim.get_data()
#         np.savetxt(path, data)

#         return sim


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
#      

#         if config.action["inital_block"] <= sim.block <= config.action["final_block"]:
#             cur_val = sim.context.getParameter(config.action["variable_name"])

#             new_val = cur_val + (
#                 (config.action["final_value"] - cur_val)
#                 / (config.action["final_block"] - sim.block + 1)
#             )

#             logging.info(f'set {config.action["variable_name"]} to {new_val}')
#             sim.context.setParameter(config.action["variable_name"], new_val)


