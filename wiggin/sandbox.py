

class AddInitConfCylindricalConfinement(SimAction):
    # TODO: redo as a configuration step?..
    def __init__(
        self,
        k=1.0,
        r_max=None,
        z_min=None,
        z_max=None,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.args))

        coords = config.shared["initial_conformation"]

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

        action_config["r_max"] = ((coords[:, :2] ** 2).sum(axis=1) ** 0.5).max()

        return newconf

    def run_init(self, config: ConfigEntry, sim):
        # do not use self.args!
        # only use parameters from action_configs[self.name] and config.shared

        sim.add_force(
            extra_forces.cylindrical_confinement_2(
                sim_object=sim,
                r=config.action["r_max"],
                top=config.action["z_max"],
                bottom=config.action["z_min"],
                k=config.action["k"],
            )
        )


class AddDynamicCylinderCompression(SimAction):
    def __init__(
        self,
        final_per_particle_volume=1.5 * 1.5 * 1.5,
        final_axial_compression=1,
        powerlaw=2.0,
        initial_block=1,
        final_block=100,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.args))

        init_bottom = action_configs["AddInitConfCylindricalConfinement"]["z_min"]
        init_top = action_configs["AddInitConfCylindricalConfinement"]["z_max"]
        init_mid = (init_top + init_bottom) / 2
        init_height = init_top - init_bottom

        final_top = (
            init_mid + init_height / 2 / action_config["final_axial_compression"]
        )

        final_bottom = (
            init_mid - init_height / 2 / action_config["final_axial_compression"]
        )

        final_r = np.sqrt(
            config.shared["N"]
            * action_config["final_per_particle_volume"]
            / (final_top - final_bottom)
            / np.pi
        )

        action_config["final_r"] = final_r
        action_config["final_top"] = final_top
        action_config["final_bottom"] = final_bottom

        return newconf

    def run_loop(self, config:ConfigEntry, sim):
        # do not use self.args!
        # only use parameters from action_configs[self.name] and config.shared

        if config.action["initial_block"] <= sim.block <= config.action["final_block"]:
            ks = [
                k for k in ["r", "top", "bottom"] if config.action[f"final_{k}"] is not None
            ]

            cur_vals = {
                k: sim.context.getParameter(f"cylindrical_confinement_{k}") for k in ks
            }

            new_vals = {
                k: cur_vals[k]
                + (
                    (cur_vals[k] - config.action[f"final_{k}"])
                    * (
                        (
                            (config.action["final_block"] + 1 - sim.block - 1)
                            ** config.action["powerlaw"]
                        )
                        / (
                            (config.action["final_block"] + 1 - sim.block)
                            ** config.action["powerlaw"]
                        )
                        - 1
                    )
                )
                for k in ks
            }

            for k in ks:
                sim.context.setParameter(
                    f"cylindrical_confinement_{k}", new_vals[k] * sim.conlen
                )

            if "AddTipsTethering" in action_configs:
                if "top" in ks and "bottom" in ks:
                    sim.force_dict["Tethers"].setParticleParameters(
                        0, 0, [0, 0, new_vals["bottom"]]
                    )
                    sim.force_dict["Tethers"].setParticleParameters(
                        1, sim.N - 1, [0, 0, new_vals["top"]]
                    )
                    sim.force_dict["Tethers"].updateParametersInContext(sim.context)


class AddTwoStepDynamicCylinderCompression(SimAction):
    def __init__(
        self,
        final_per_particle_volume=1.5 * 1.5 * 1.5,
        final_axial_compression=1,
        powerlaw=2.0,
        step1_start=1,
        step1_end=100,
        step2_start=100,
        step2_end=200,
    ):
        
        super().__init__(**locals())

    def configure(self, config: ConfigEntry):
        newconf = ConfigEntry(shared={}, action=copy.deepcopy(self.args))

        init_bottom = action_configs["AddInitConfCylindricalConfinement"]["z_min"]
        init_top = action_configs["AddInitConfCylindricalConfinement"]["z_max"]
        init_mid = (init_top + init_bottom) / 2
        init_height = init_top - init_bottom

        step1_top = init_top
        step1_bottom = init_bottom
        step1_r = np.sqrt(
            config.shared["N"]
            * action_config["final_per_particle_volume"]
            / (step1_top - step1_bottom)
            / np.pi
        )

        step2_top = (
            init_mid + init_height / 2 / action_config["final_axial_compression"]
        )
        step2_bottom = (
            init_mid - init_height / 2 / action_config["final_axial_compression"]
        )
        step2_r = np.sqrt(
            config.shared["N"]
            * action_config["final_per_particle_volume"]
            / (step2_top - step2_bottom)
            / np.pi
        )

        action_config["step1_r"] = step1_r
        action_config["step1_top"] = step1_top
        action_config["step1_bottom"] = step1_bottom

        action_config["step2_r"] = step2_r
        action_config["step2_top"] = step2_top
        action_config["step2_bottom"] = step2_bottom

        return newconf

    def run_loop(self, config:ConfigEntry, sim):
        # do not use self.args!
        # only use parameters from action_configs[self.name] and config.shared

        if (config.action["step1_start"] <= sim.block <= config.action["step1_end"]) or (
            config.action["step2_start"] <= sim.block <= config.action["step2_end"]
        ):
            step = (
                "step1"
                if (config.action["step1_start"] <= sim.block <= config.action["step1_end"])
                else "step2"
            )
            ks = [
                k
                for k in ["r", "top", "bottom"]
                if config.action[f"{step}_{k}"] is not None
            ]

            cur_vals = {
                k: sim.context.getParameter(f"cylindrical_confinement_{k}") for k in ks
            }

            new_vals = {
                k: cur_vals[k]
                + (
                    (cur_vals[k] - config.action[f"{step}_{k}"])
                    * (
                        (
                            (config.action[f"{step}_end"] + 1 - sim.block - 1)
                            ** config.action["powerlaw"]
                        )
                        / (
                            (config.action[f"{step}_end"] + 1 - sim.block)
                            ** config.action["powerlaw"]
                        )
                        - 1
                    )
                )
                for k in ks
            }

            for k in ks:
                sim.context.setParameter(
                    f"cylindrical_confinement_{k}", new_vals[k] * sim.conlen
                )

            if "AddTipsTethering" in action_configs:
                if "top" in ks and "bottom" in ks:
                    sim.force_dict["Tethers"].setParticleParameters(
                        0, 0, [0, 0, new_vals["bottom"]]
                    )
                    sim.force_dict["Tethers"].setParticleParameters(
                        1, sim.N - 1, [0, 0, new_vals["top"]]
                    )
                    sim.force_dict["Tethers"].updateParametersInContext(sim.context)

