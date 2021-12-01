import copy
import logging
import numbers

import numpy as np

import looplib
import looplib.looptools

from polychrom import forces, forcekits

from .simconstructor import SimAction, ConfigEntry

from . import starting_mitotic_conformations, extra_forces

logging.basicConfig(level=logging.INFO)


class GenerateSingleLayerLoops(SimAction):
    def __init__(
        self,
        loop_size=400,
        loop_gamma_k=1,
        loop_spacing=1,
        chain_idxs=None,
    ):
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        import looplib
        import looplib.looptools
        import looplib.random_loop_arrays

        if self.params["chain_idxs"] is None:
            if "loop_n" in self.params:
                N = self.params["loop_n"] * self.params["loop_size"]
                newconf.shared["N"] = N
            else:
                N = config.shared["N"]
            chains = [(0, N, False)]

        else:
            if "chains" not in config.shared:
                raise ValueError("Chains are not configured!")
            if hasattr(self.params["chain_idxs"], "__iter__"):
                chains = [
                    config.shared["chains"][i] for i in self.params["chain_idxs"]
                ]
            else:
                chains = [config.shared["chains"][int(self.params["chain_idxs"])]]

        loops = []
        for start, end, is_ring in chains:
            chain_len = end - start
            if self.params["loop_gamma_k"] == 1:
                loops.append(
                    looplib.random_loop_arrays.exponential_loop_array(
                        chain_len,
                        self.params["loop_size"],
                        self.params["loop_spacing"]
                    )
                )
            else:
                loops.append(
                    looplib.random_loop_arrays.gamma_loop_array(
                        chain_len,
                        self.params["loop_size"],
                        self.params["loop_gamma_k"],
                        self.params["loop_spacing"],
                        min_loop_size=3
                    )
                )
            loops[0] += start
        loops = np.vstack(loops)

        newconf.shared["loops"] = (
            loops
            if "loops" not in newconf.shared
            else np.vstack([newconf.shared["loops"], loops])
        )

        try:
            newconf.shared["backbone"] = looplib.looptools.get_backbone(
                newconf.shared["loops"], N=N
            )
        except Exception:
            newconf.shared["backbone"] = None

        return newconf


class GenerateRandomParticleTypes(SimAction):
    def __init__(
        self,
        freqs=[0.5, 0.5],
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        freqs_norm = np.array(self.params["freqs"])
        newconf.action["freqs_norm"] /= freqs_norm.sum()

        newconf.shared["particle_types"] = np.random.choice(
            a=np.arange(freqs_norm.size), size=config.shared["N"], p=freqs_norm
        )

        return newconf


class GenerateRandomBlockParticleTypes(SimAction):
    def __init__(
        self,
        avg_block_lens=[2, 2],
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        # This solution is slow-ish (1 sec for 1e6 particles), but simple
        N = config.shared["N"]
        avg_block_lens = self.params["avg_block_lens"]
        n_types = len(avg_block_lens)
        particle_types = np.full(N, -1)

        p, new_p, t = 0, 0, 0
        while new_p <= N:
            new_p = p + np.random.geometric(1 / avg_block_lens[t])
            particle_types[p : min(new_p, N)] = t
            t = (t + 1) % n_types
            p = new_p

        newconf.shared["particle_types"] = particle_types

        return newconf


class GenerateTwoLayerLoops(SimAction):
    def __init__(
        self,
        inner_loop_size=400,
        outer_loop_size=400 * 4,
        inner_loop_spacing=1,
        outer_loop_spacing=1,
        outer_inner_offset=1,
        inner_loop_gamma_k=1,
        outer_loop_gamma_k=1,
    ):        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        import looplib
        import looplib.looptools
        import looplib.random_loop_arrays

        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        if "outer_loop_n" in self.params:
            N = self.params["outer_loop_n"] * self.params["outer_loop_size"]
            newconf.shared["N"] = N
        elif "inner_loop_n" in self.params:
            N = self.params["inner_loop_n"] * self.params["inner_loop_size"]
            newconf.shared["N"] = N
        else:
            N = config.shared["N"]

        (
            outer_loops,
            inner_loops,
        ) = looplib.random_loop_arrays.two_layer_gamma_loop_array(
            N,
            self.params["outer_loop_size"],
            self.params["outer_loop_gamma_k"],
            self.params["outer_loop_spacing"],
            self.params["inner_loop_size"],
            self.params["inner_loop_gamma_k"],
            self.params["inner_loop_spacing"],
            self.params["outer_inner_offset"],
        )
        loops = np.vstack([outer_loops, inner_loops])
        loops.sort()

        newconf.shared["loops"] = (
            loops
            if "loops" not in newconf.shared
            else np.vstack([newconf.shared["loops"], loops])
        )

        action_config["inner_loops"] = inner_loops
        action_config["outer_loops"] = outer_loops

        newconf.shared["backbone"] = looplib.looptools.get_backbone(
            outer_loops, N=N
        )

        return newconf


class AddLoops(SimAction):
    def __init__(
        self,
        wiggle_dist=0.05,
        bond_length=1.0,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=config.shared["loops"],
                bondWiggleDistance=config.action["wiggle_dist"],
                bondLength=config.action["bond_length"],
                name="loop_harmonic_bonds",
                override_checks=True,
            )
        )


class AddRootLoopSeparator(SimAction):
    def __init__(
        self,
        wiggle_dist=0.1,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        loops = config.shared["loops"]
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_spacers = np.vstack([root_loops[:-1][:, 1], root_loops[1:][:, 0]]).T
        root_loop_spacer_lens = root_loop_spacers[:, 1] - root_loop_spacers[:, 0]

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=root_loop_spacers,
                bondWiggleDistance=config.action["wiggle_dist"],
                bondLength=root_loop_spacer_lens,
                name="RootLoopSpacers",
                override_checks=True,
            )
        )


class AddBackboneStiffness(SimAction):
    def __init__(
        self,
        k=1.5,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        bb_list = sorted(config.shared["backbone"])
        triplets = [bb_list[i : i + 3] for i in range(len(bb_list) - 2)]
        sim.add_force(
            forces.angle_force(
                sim_object=sim,
                triplets=triplets,
                k=config.action["k"],
                theta_0=np.pi,
                name="backbone_stiffness",
                override_checks=True,
            )
        )


class AddBackboneTethering(SimAction):
    def __init__(
        self,
        k=15,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            forces.tether_particles(
                sim_object=sim,
                particles=config.shared["backbone"],
                k=config.action["k"],
                positions="current",
                name="tether_backbone",
            )
        )


class AddBackboneAngularTethering(SimAction):
    def __init__(
        self,
        angle_wiggle=np.pi / 16,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            extra_forces.angular_tether_particles(
                sim_object=sim,
                particles=config.shared["backbone"],
                angle_wiggle=config.action["angle_wiggle"],
                angles="current",
                name="tether_backbone_angle",
            )
        )


class AddRootLoopAngularTethering(SimAction):
    def __init__(
        self,
        angle_wiggle=np.pi / 16,
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        loops = config.shared["loops"]
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_particles = sorted(np.unique(root_loops))

        sim.add_force(
            extra_forces.angular_tether_particles(
                sim_object=sim,
                particles=root_loop_particles,
                angle_wiggle=config.action["angle_wiggle"],
                angles="current",
                name="tether_root_loops_angle",
            )
        )


class AddTipsTethering(SimAction):
    def __init__(
        self,
        k=(0, 0, 5),
        particles=(0, -1),
        positions="current",
    ):
        
        super().__init__(**locals())

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            forces.tether_particles(
                sim_object=sim,
                particles=config.action["particles"],
                k=config.action["k"],
                positions=config.action["positions"],
            )
        )


class AddStaticCylinderCompression(SimAction):
    def __init__(
        self, k=1.0, z_min=None, z_max=None, r=None, per_particle_volume=1.5 * 1.5 * 1.5
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        if (action_config["z_min"] is None) != (action_config["z_max"] is None):
            raise ValueError(
                "Both z_min and z_max have to be either specified or left as None."
            )

        coords = config.shared["initial_conformation"]
        
        if action_config["z_min"] is None:
            action_config["z_min"] = coords[:, 2].min()
        elif action_config["z_min"] == "bb":
            action_config["z_min"] = coords[config.shared["backbone"]][:, 2].min()
        else:
            action_config["z_min"] = action_config["z_min"]

        if action_config["z_max"] is None:
            action_config["z_max"] = coords[:, 2].max()
        elif action_config["z_max"] == "bb":
            action_config["z_max"] = coords[config.shared["backbone"]][:, 2].max()
        else:
            action_config["z_max"] = action_config["z_max"]

        if (action_config["r"] is not None) and (
            action_config["per_particle_volume"] is not None
        ):
            raise ValueError("Please specify either r or per_particle_volume.")
        elif (action_config["r"] is None) and (
            action_config["per_particle_volume"] is None
        ):
            coords = config.shared["initial_conformation"]
            action_config["r"] = ((coords[:, :2] ** 2).sum(axis=1) ** 0.5).max()
        elif (action_config["r"] is None) and (
            action_config["per_particle_volume"] is not None
        ):
            action_config["r"] = np.sqrt(
                config.shared["N"]
                * action_config["per_particle_volume"]
                / (action_config["z_max"] - action_config["z_min"])
                / np.pi
            )

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            extra_forces.cylindrical_confinement_2(
                sim_object=sim,
                r=config.action["r"],
                top=config.action["z_max"],
                bottom=config.action["z_min"],
                k=config.action["k"],
            )
        )


class GenerateLoopBrushInitialConformation(SimAction):
    def __init__(
        self,
        helix_radius=None,
        helix_turn_length=None,
        helix_step=None,
        axial_compression_factor=None,
        random_loop_orientations=True,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        n_params = sum(
            [
                i is None
                for i in [
                    action_config["helix_radius"],
                    action_config["helix_turn_length"],
                    action_config["helix_step"],
                    action_config["axial_compression_factor"],
                ]
            ]
        )

        if n_params not in [0, 2]:
            raise ValueError(
                "Please specify 0 or 2 out of these four parameters: "
                "radius, turn_length, step and axis-to-backbone ratio"
            )

        if (action_config["helix_radius"] is not None) and (
            action_config["helix_step"] is not None
        ):
            helix_radius = action_config["helix_radius"]
            helix_step = action_config["helix_step"]
        elif (action_config["helix_turn_length"] is not None) and (
            action_config["helix_step"] is not None
        ):
            helix_step = action_config["helix_step"]
            helix_radius_squared = (
                (
                    (action_config["helix_turn_length"]) ** 2
                    - (action_config["helix_step"]) ** 2
                )
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5

        elif (action_config["helix_turn_length"] is not None) and (
            action_config["helix_radius"] is not None
        ):
            helix_radius = action_config["helix_radius"]
            helix_step_squared = (action_config["helix_turn_length"]) ** 2 - (
                2 * np.pi * helix_radius
            ) ** 2
            if helix_step_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_step = helix_step_squared ** 0.5

        elif (action_config["axial_compression_factor"] is not None) and (
            action_config["helix_radius"] is not None
        ):
            helix_radius = action_config["helix_radius"]
            helix_step = (
                2
                * np.pi
                * helix_radius
                / np.sqrt(action_config["axial_compression_factor"] ** 2 - 1)
            )

        elif (action_config["axial_compression_factor"] is not None) and (
            action_config["helix_turn_length"] is not None
        ):
            helix_step = (
                action_config["helix_turn_length"]
                / action_config["axial_compression_factor"]
            )
            helix_radius_squared = (
                ((action_config["helix_turn_length"]) ** 2 - (helix_step) ** 2)
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5
        elif (action_config["axial_compression_factor"] is not None) and (
            action_config["helix_step"] is not None
        ):
            helix_step = action_config["helix_step"]
            helix_turn_length = helix_step * action_config["axial_compression_factor"]
            helix_radius_squared = (
                ((helix_turn_length) ** 2 - (helix_step) ** 2)
                / np.pi
                / np.pi
                / 2.0
                / 2.0
            )
            if helix_radius_squared <= 0:
                raise ValueError(
                    "The provided values of helix_step and helix_turn_length are incompatible"
                )
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        action_config["helix_step"] = helix_step
        action_config["helix_radius"] = helix_radius

        newconf.shared[
            "initial_conformation"
        ] = starting_mitotic_conformations.make_helical_loopbrush(
            L=config.shared["N"],
            helix_radius=helix_radius,
            helix_step=helix_step,
            loops=config.shared["loops"],
            random_loop_orientations=action_config["random_loop_orientations"],
        )

        return newconf


class GenerateLoopBrushUniformHelixInitialConformation(SimAction):
    def __init__(
        self,
        helix_radius=None,
        helix_step=None,
        axial_compression_factor=None,
        period_particles=None,
        loop_fold="RW",
        loop_layer="all",
        chain_bond_length=1.0,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        n_params = sum(
            [
                i is not None
                for i in [
                    action_config["helix_radius"],
                    action_config["helix_step"],
                    action_config["axial_compression_factor"],
                ]
            ]
        )

        if n_params not in [0, 2]:
            raise ValueError(
                "Please specify 0 or 2 out of these four parameters: "
                "radius, step and axis-to-backbone ratio"
            )

        if (action_config["helix_radius"] is not None) and (
            action_config["helix_step"] is not None
        ):
            helix_radius = action_config["helix_radius"]
            helix_step = action_config["helix_step"]
        elif (action_config["axial_compression_factor"] is not None) and (
            action_config["helix_radius"] is not None
        ):
            helix_radius = action_config["helix_radius"]
            helix_step = (
                2
                * np.pi
                * helix_radius
                / np.sqrt(action_config["axial_compression_factor"] ** 2 - 1)
            )

        elif (action_config["axial_compression_factor"] is not None) and (
            action_config["helix_step"] is not None
        ):
            helix_step = action_config["helix_step"]
            helix_turn_length = helix_step * action_config["axial_compression_factor"]
            helix_radius_squared = (
                (helix_turn_length ** 2 - helix_step ** 2) / np.pi / np.pi / 2.0 / 2.0
            )
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        action_config["helix_step"] = helix_step
        action_config["helix_radius"] = helix_radius

        newconf.shared[
            "initial_conformation"
        ] = starting_mitotic_conformations.make_uniform_helical_loopbrush(
            L=config.shared["N"],
            helix_radius=helix_radius,
            helix_step=helix_step,
            period_particles=action_config["period_particles"],
            loops=config.shared["loops"],
            chain_bond_length=action_config["chain_bond_length"],
            loop_fold=action_config["loop_fold"],
        )

        return newconf




class AddChainsSelectiveRepAttr(SimAction):
    def __init__(
        self,
        chains=((0, None, 0),),
        bond_length=1.0,
        wiggle_dist=0.025,
        stiffness_k=None,
        repulsion_e=2.5,  # TODO: implement np.inf
        attraction_e=None,
        attraction_r=None,
        selective_attraction_e=None,
        particle_types=None,
        except_bonds=False,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        if hasattr(action_config["chains"], "__iter__") and hasattr(
            action_config["chains"][0], "__iter__"
        ):
            newconf.shared["chains"] = action_config["chains"]
        elif hasattr(action_config["chains"], "__iter__") and isinstance(
            action_config["chains"][0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(action_config["chains"])]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            action_config["chains"] = chains
            newconf.shared["chains"] = chains

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        nonbonded_force_func = extra_forces.homotypic_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
            repulsionEnergy=config.action["repulsion_e"],
            repulsionRadius=1.0,
            attractionEnergy=config.action["attraction_e"],
            attractionRadius=config.action["attraction_r"],
            particleTypes=config.action["particle_types"],
            selectiveAttractionEnergy=config.action["selective_attraction_e"],
        )

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=config.shared["chains"],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": config.action["bond_length"],
                    "bondWiggleDistance": config.action["wiggle_dist"],
                },
                angle_force_func=(
                    None if config.action["stiffness_k"] is None else forces.angle_force
                ),
                angle_force_kwargs={"k": config.action["stiffness_k"]},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=config.action["except_bonds"],
            )
        )


class AddChainsHeteropolymerRepAttr(SimAction):
    def __init__(
        self,
        chains=((0, None, 0),),
        bond_length=1.0,
        wiggle_dist=0.025,
        stiffness_k=None,
        repulsion_e=2.5,  # TODO: implement np.inf
        attraction_e=None,
        attraction_r=None,
        particle_types=None,
        except_bonds=False,
    ):
        
        params["chains"] = np.array(chains, dtype=np.object)
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.params))

        if hasattr(action_config["chains"], "__iter__") and hasattr(
            action_config["chains"][0], "__iter__"
        ):
            newconf.shared["chains"] = action_config["chains"]
        elif hasattr(action_config["chains"], "__iter__") and isinstance(
            action_config["chains"][0], numbers.Number
        ):
            edges = np.r_[0, np.cumsum(action_config["chains"])]
            chains = np.array(
                [(st, end, False) for st, end in zip(edges[:-1], edges[1:])],
                dtype=np.object,
            )
            action_config["chains"] = chains
            newconf.shared["chains"] = chains

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and config.shared

        nonbonded_force_func = extra_forces.heteropolymer_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
            repulsionEnergy=config.action["repulsion_e"],
            repulsionRadius=1.0,
            attractionEnergies=config.action["attraction_e"],
            attractionRadius=config.action["attraction_r"],
            particleTypes=(
                config.shared["particle_types"]
                if config.action["particle_types"] is None
                else config.action["particle_types"]
            ),
        )

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=config.shared["chains"],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    "bondLength": config.action["bond_length"],
                    "bondWiggleDistance": config.action["wiggle_dist"],
                },
                angle_force_func=(
                    None if config.action["stiffness_k"] is None else forces.angle_force
                ),
                angle_force_kwargs={"k": config.action["stiffness_k"]},
                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,
                except_bonds=config.action["except_bonds"],
            )
        )
