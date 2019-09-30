import os, shelve

import numpy as np

from polychrom import simulation, forces, forcekits

from simconstructor import AttrDict, SimulationAction

import starting_mitotic_conformations


class GenerateSingleLayerLoops(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        loop_size = 200,
#        loop_gamma_k = 1,
        loop_spacing = 1,
    )

    def configure(self, shared_config, action_configs):
        import looplib
        import looplib.looptools
        import looplib.random_loop_arrays

        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if 'loop_n' in self.params:
            N = self.params['loop_n'] * action_config['loop_size']
            shared_config_added_data['N'] = N
        else:
            N = shared_config['N']
               
        # TODO: move to share
        shared_config_added_data['loops'] = (
            looplib.random_loop_arrays.exponential_loop_array(
                N, 
                action_config['loop_size'],
                action_config['loop_spacing'])
        )

        shared_config_added_data['backbone'] = looplib.looptools.get_backbone(
                shared_config_added_data['loops'], N=N)

        return shared_config_added_data, action_config


class GenerateTwoLayerLoops(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        inner_loop_size = 200,
        outer_loop_size = 200 * 4,
        inner_loop_spacing = 1,
        outer_loop_spacing = 1,
        inner_outer_spacing = 1,
        inner_loop_gamma_k = 1,
        outer_loop_gamma_k = 1,
    )


    def configure(self, shared_config, action_configs):
        import looplib
        import looplib.looptools
        import looplib.random_loop_arrays

        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if 'outer_loop_n' in self.params:
            N = self.params['outer_loop_n'] * action_config['outer_loop_size']
            shared_config_added_data['N'] = N
        elif 'inner_loop_n' in self.params:
            N = self.params['inner_loop_n'] * action_config['inner_loop_size']
            shared_config_added_data['N'] = N
        else:
            N = shared_config['N']
               
        outer_loops, inner_loops = two_layer_gamma_loop_array(N,
                          outer_loop_size, outer_gamma_k, outer_loop_spacing,
                          inner_loop_size, inner_gamma_k, inner_loop_spacing,
                          outer_inner_offset=1)
        loops = sorted(outer_loops+inner_loops, key=lambda x: x[0])

        shared_config_added_data['loops'] = loops
        action_config['inner_loops'] = inner_loops
        action_config['outer_loops'] = outer_loops

        shared_config_added_data['backbone'] = looplib.looptools.get_backbone(
                outer_loops, N)

        return shared_config_added_data, action_config


class AddLoops(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        wiggle_dist=0.05,
        bond_length=1.0
    )

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=shared_config['loops'],
                bondWiggleDistance=self_conf.wiggle_dist,
                bondLength=self_conf.bond_length,
                name='LoopHarmonicBonds'
            )
        )

        return sim


class AddInitConfCylindricalConfinement(SimulationAction):
    # TODO: redo as a configuration step?..
    stage = 'init'
    _default_params = AttrDict(
        k=1.0,
        r_max=None,
        z_min=None,
        z_max=None,
    )

    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        coords = shared_config['initial_conformation']

        action_config['r_max'] = ((coords[:,:2]**2).sum(axis=1)**0.5).max()
        action_config['z_min'] = coords[:,2].min()
        action_config['z_max'] = coords[:,2].max()

        return shared_config_added_data, action_config

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.cylindrical_confinement(
                sim_object=sim,
                r=self_conf.r_max,
                top=self_conf.z_max,
                bottom=self_conf.z_min, 
                k=self_conf.k
            )
        )

        return sim


class AddTipsTethering(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        k=[0,0,5],
        particles=[0, -1],
        positions='current',
    )


    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.tether_particles(
                sim_object=sim, 
                particles=self_conf.particles, 
                k=self_conf.k, 
                positions=self_conf.positions,
            )
        )

        return sim


class SaveConfiguration(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        backup = True
    )

    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        os.mkdir(shared_config['folder'])
        paths = [os.path.join(shared_config['folder'], 'conf.shlv')] 

        if action_config['backup']:
            os.mkdir(shared_config['folder']+'/backup/')
            paths.append(os.path.join(shared_config['folder'], 'backup', 'conf.shlv'))

        for path in paths:
            conf = shelve.open(path, protocol=2)
            # TODO: fix saving action_params
            # conf['params'] = {k:dict(v) for k,v in ..}
            conf['shared_config'] = shared_config
            conf['action_configs'] = {k:dict(v) for k,v in action_configs.items()}
            conf.close()

        return shared_config_added_data, action_config


class AddDynamicCylinderCompression(SimulationAction):
    stage = 'loop'
    _default_params = AttrDict(
        final_per_particle_volume = 1.5*1.5*1.5,
        final_axial_compression = 1,
        powerlaw = 1.0,
        initial_block = 0,
        final_block = 100,
    )

    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        init_bottom = action_configs.AddInitConfCylindricalConfinement.z_min
        init_top = action_configs.AddInitConfCylindricalConfinement.z_max
        init_mid = (init_top + init_bottom) / 2
        init_height = (init_top - init_bottom)

        final_top = init_mid + init_height / 2 / action_config.final_axial_compression
        final_bottom = init_mid - init_height / 2 / action_config.final_axial_compression

        final_r = np.sqrt(
            shared_config.N * action_config.final_per_particle_volume 
            / (final_top - final_bottom) / np.pi
        )

        action_config['final_r'] = final_r 
        action_config['final_top'] = final_top
        action_config['final_bottom'] = final_bottom

        return shared_config_added_data, action_config

    def run(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]


        if self_conf.initial_block <= sim.block <= self_conf.final_block:
            ks = [k for k in ['r', 'top', 'bottom']
                  if self_conf[f'final_{k}'] is not None]

            cur_vals = {
                k:sim.context.getParameter(f'cylindrical_confinement_{k}')
                for k in ks
            }

            new_vals = {
                k:cur_vals[k] + (
                    (cur_vals[k] - self_conf[f'final_{k}']) 
                    * (
                        ((self_conf.final_block + 1 - sim.block - 1) ** self_conf.powerlaw)
                        / ((self_conf.final_block + 1 - sim.block ) ** self_conf.powerlaw)
                        - 1
                    )
                )
                for k in ks
            }

            for k in ks:
                sim.context.setParameter(
                    f'cylindrical_confinement_{k}', new_vals[k] * sim.nm)

            if 'AddTipsTethering' in action_configs:
                if 'top' in ks and 'bottom' in ks:
                    sim.force_dict['Tethers'].setParticleParameters(
                        0, 0, [0, 0, new_vals['bottom']])
                    sim.force_dict['Tethers'].setParticleParameters(
                        1, sim.N-1, [0, 0, new_vals['top']])
                    sim.force_dict['Tethers'].updateParametersInContext(
                        sim.context)

        return sim


class GenerateLoopBrushInitialConformation(SimulationAction):
    stage = 'init'
    _default_params = AttrDict(
        helix_radius=0,
        helix_step=1000000,
        random_loop_orientations=True,
    )

    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)
    
        shared_config_added_data.initial_conformation = (
            starting_mitotic_conformations.make_helical_loopbrush(
                L=shared_config.N,
                helix_radius=action_config.helix_radius,
                helix_step=action_config.helix_step,
                loops=shared_config.loops,
                random_loop_orientations=action_config.random_loop_orientations
            )
        )

        return shared_config_added_data, action_config

