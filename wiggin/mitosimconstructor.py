import os
import sys
import shelve
import logging
import numbers

import numpy as np

import looplib
import looplib.looptools

from polychrom import forces, forcekits

from .simconstructor import SimAction

from . import starting_mitotic_conformations, extra_forces

logging.basicConfig(level=logging.INFO)


class GenerateSingleLayerLoops(SimAction):
    def __init__(
        self,
        loop_size=400,
        # loop_gamma_k = 1,
        loop_spacing=1,
        chain_idxs=None,
    ):
        params = {k: v for k, v in locals().items() if k not in ['self']}  # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        import looplib
        import looplib.looptools
        import looplib.random_loop_arrays

        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if action_config['chain_idxs'] is None:
            if 'loop_n' in self.params:
                N = self.params['loop_n'] * action_config['loop_size']
                shared_config_added_data['N'] = N
            else:
                N = shared_config['N']
            chains = [(0, N, False)]
            
        else:
            if 'chains' not in shared_config:
                raise ValueError('Chains are not configured!')
            if hasattr(action_config['chain_idxs'], '__iter__'):
                chains = [shared_config['chains'][i] for i in action_config['chain_idxs']]
            else:
                chains = [shared_config['chains'][int(action_config['chain_idxs'])]]
               
        loops = []
        for start, end, is_ring in chains:
            chain_len = end-start
            loops.append(
                looplib.random_loop_arrays.exponential_loop_array(
                    chain_len, 
                    action_config['loop_size'],
                    action_config['loop_spacing']
            ))
            loops[0] += start
        loops = np.vstack(loops)

        shared_config_added_data['loops'] = (
            loops 
            if 'loops' not in shared_config_added_data
            else np.vstack([shared_config_added_data['loops'], loops])
        )
                
        try:
            shared_config_added_data['backbone'] = looplib.looptools.get_backbone(
                    shared_config_added_data['loops'], N=N)
        except:
            shared_config_added_data['backbone'] = None

        return shared_config_added_data, action_config


class GenerateRandomParticleTypes(SimAction):
    def __init__(
        self,
        freqs=[0.5, 0.5],
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        freqs_norm = np.array(action_config['freqs'])
        action_config['freqs_norm'] /= freqs_norm.sum()
       
        shared_config_added_data['particle_types'] = np.random.choice(
            a=np.arange(freqs_norm.size),
            size=shared_config['N'],
            p=freqs_norm
        )

        return shared_config_added_data, action_config


class GenerateRandomBlockParticleTypes(SimAction):
    def __init__(
        self,
        avg_block_lens=[2,2],
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        # This solution is slow-ish (1 sec for 1e6 particles), but simple
        N = shared_config['N']
        avg_block_lens = action_config['avg_block_lens']
        n_types = len(avg_block_lens)
        particle_types = np.full(N, -1)

        p, new_p, t = 0, 0, 0
        while new_p <= N:
            new_p = p + np.random.geometric(1/avg_block_lens[t])
            particle_types[p:min(new_p,N)] = t
            t = (t+1) % n_types
            p = new_p
            
        shared_config_added_data['particle_types'] = particle_types

        return shared_config_added_data, action_config


class GenerateTwoLayerLoops(SimAction):
    def __init__(
        self,
        inner_loop_size = 400,
        outer_loop_size = 400 * 4,
        inner_loop_spacing = 1,
        outer_loop_spacing = 1,
        outer_inner_offset= 1,
        inner_loop_gamma_k = 1,
        outer_loop_gamma_k = 1,
    ):

        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)

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
               
        outer_loops, inner_loops = looplib.random_loop_arrays.two_layer_gamma_loop_array(
            N,
            action_config['outer_loop_size'], 
            action_config['outer_loop_gamma_k'], 
            action_config['outer_loop_spacing'],
            action_config['inner_loop_size'], 
            action_config['inner_loop_gamma_k'], 
            action_config['inner_loop_spacing'],
            action_config['outer_inner_offset'])
        loops = np.vstack([outer_loops, inner_loops])
        loops.sort()

        shared_config_added_data['loops'] = (
            loops 
            if 'loops' not in shared_config_added_data
            else np.vstack([shared_config_added_data['loops'], loops])
        )

        action_config['inner_loops'] = inner_loops
        action_config['outer_loops'] = outer_loops

        shared_config_added_data['backbone'] = looplib.looptools.get_backbone(
                outer_loops, N=N)

        return shared_config_added_data, action_config


class AddLoops(SimAction):
    def __init__(
        self,
        wiggle_dist=0.05,
        bond_length=1.0,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=shared_config['loops'],
                bondWiggleDistance=self_conf['wiggle_dist'],
                bondLength=self_conf['bond_length'],
                name='loop_harmonic_bonds',
                override_checks=True
            )
        )



class AddRootLoopSeparator(SimAction):
    def __init__(
        self,
        wiggle_dist=0.1,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        loops = shared_config['loops']
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_spacers = np.vstack([root_loops[:-1][:,1], root_loops[1:][:,0]]).T
        root_loop_spacer_lens = root_loop_spacers[:,1] - root_loop_spacers[:,0]

        sim.add_force(
            forces.harmonic_bonds(
                sim_object=sim,
                bonds=root_loop_spacers,
                bondWiggleDistance=self_conf['wiggle_dist'],
                bondLength=root_loop_spacer_lens,
                name='RootLoopSpacers',
                override_checks=True
            )
        )


class AddBackboneStiffness(SimAction):
    def __init__(
        self,
        k=1.5,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)

    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        bb_list = sorted(shared_config['backbone'])
        triplets = [bb_list[i:i+3] for i in range(len(bb_list)-2)]
        sim.add_force(
            forces.angle_force(
                sim_object=sim, 
                triplets=triplets, 
                k=self_conf['k'], 
                theta_0=np.pi, 
                name="backbone_stiffness", 
                override_checks=True
            )
        )


class AddBackboneTethering(SimAction):
    def __init__(
        self,
        k=15,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.tether_particles(
                sim_object=sim, 
                particles=shared_config['backbone'], 
                k=self_conf['k'],
                positions='current',
                name='tether_backbone'
            )
        )
        

class AddBackboneAngularTethering(SimAction):
    def __init__(
        self,
        angle_wiggle=np.pi/16,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            extra_forces.angular_tether_particles(
                sim_object=sim, 
                particles=shared_config['backbone'], 
                angle_wiggle=self_conf['angle_wiggle'],
                angles='current',
                name='tether_backbone_angle'
            )
        )
        

class AddRootLoopAngularTethering(SimAction):
    def __init__(
        self,
        angle_wiggle=np.pi/16,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        loops = shared_config['loops']
        root_loops = loops[looplib.looptools.get_roots(loops)]
        root_loop_particles = sorted(np.unique(root_loops))

        sim.add_force(
            extra_forces.angular_tether_particles(
                sim_object=sim, 
                particles=root_loop_particles, 
                angle_wiggle=self_conf['angle_wiggle'],
                angles='current',
                name='tether_root_loops_angle'
            )
        )
        


class AddTipsTethering(SimAction):
    def __init__(
        self,
        k=(0,0,5),
        particles=(0, -1),
        positions='current',
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)



    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            forces.tether_particles(
                sim_object=sim, 
                particles=self_conf['particles'], 
                k=self_conf['k'], 
                positions=self_conf['positions'],
            )
        )


class AddInitConfCylindricalConfinement(SimAction):
    # TODO: redo as a configuration step?..
    def __init__(
        self,
        k=1.0,
        r_max=None,
        z_min=None,
        z_max=None,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        coords = shared_config['initial_conformation']

        if ((action_config['z_min'] is None) != (action_config['z_max'] is None)):
            raise ValueError('Both z_min and z_max have to be either specified or left as None.')
        coords = shared_config['initial_conformation']
        if (action_config['z_min'] is None):
            action_config['z_min'] = coords[:,2].min()
        elif (action_config['z_min'] == 'bb'):
            action_config['z_min'] = coords[shared_config['backbone']][:,2].min()
        else:
            action_config['z_min'] = action_config['z_min']
            
            
        if (action_config['z_max'] is None):
            action_config['z_max'] = coords[:,2].max()
        elif (action_config['z_max'] == 'bb'):
            action_config['z_max'] = coords[shared_config['backbone']][:,2] .max()
        else:
            action_config['z_max'] = action_config['z_max']

        action_config['r_max'] = ((coords[:,:2]**2).sum(axis=1)**0.5).max()

        return shared_config_added_data, action_config

    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            extra_forces.cylindrical_confinement_2(
                sim_object=sim,
                r=self_conf['r_max'],
                top=self_conf['z_max'],
                bottom=self_conf['z_min'], 
                k=self_conf['k']
            )
        )


class AddDynamicCylinderCompression(SimAction):
    def __init__(
        self,
        final_per_particle_volume = 1.5*1.5*1.5,
        final_axial_compression = 1,
        powerlaw = 2.0,
        initial_block = 1,
        final_block = 100,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        init_bottom = action_configs['AddInitConfCylindricalConfinement']['z_min']
        init_top = action_configs['AddInitConfCylindricalConfinement']['z_max']
        init_mid = (init_top + init_bottom) / 2
        init_height = (init_top - init_bottom)

        final_top = init_mid + init_height / 2 / action_config['final_axial_compression']
        final_bottom = init_mid - init_height / 2 / action_config['final_axial_compression']

        final_r = np.sqrt(
            shared_config['N'] * action_config['final_per_particle_volume'] 
            / (final_top - final_bottom) / np.pi
        )

        action_config['final_r'] = final_r 
        action_config['final_top'] = final_top
        action_config['final_bottom'] = final_bottom

        return shared_config_added_data, action_config


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if self_conf['initial_block'] <= sim.block <= self_conf['final_block']:
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
                        ((self_conf['final_block'] + 1 - sim.block - 1) ** self_conf['powerlaw'])
                        / ((self_conf['final_block'] + 1 - sim.block ) ** self_conf['powerlaw'])
                        - 1
                    )
                )
                for k in ks
            }

            for k in ks:
                sim.context.setParameter(
                    f'cylindrical_confinement_{k}', new_vals[k] * sim.conlen)

            if 'AddTipsTethering' in action_configs:
                if 'top' in ks and 'bottom' in ks:
                    sim.force_dict['Tethers'].setParticleParameters(
                        0, 0, [0, 0, new_vals['bottom']])
                    sim.force_dict['Tethers'].setParticleParameters(
                        1, sim.N-1, [0, 0, new_vals['top']])
                    sim.force_dict['Tethers'].updateParametersInContext(
                        sim.context)


class AddTwoStepDynamicCylinderCompression(SimAction):
    def __init__(
        self,
        final_per_particle_volume = 1.5*1.5*1.5,
        final_axial_compression = 1,
        powerlaw = 2.0,
        step1_start = 1,
        step1_end = 100,
        step2_start = 100,
        step2_end = 200
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        init_bottom = action_configs['AddInitConfCylindricalConfinement']['z_min']
        init_top = action_configs['AddInitConfCylindricalConfinement']['z_max']
        init_mid = (init_top + init_bottom) / 2
        init_height = (init_top - init_bottom)

        step1_top = init_top
        step1_bottom = init_bottom
        step1_r = np.sqrt(
            shared_config['N'] * action_config['final_per_particle_volume'] 
            / (step1_top - step1_bottom) / np.pi
        )
        
        step2_top = init_mid + init_height / 2 / action_config['final_axial_compression']
        step2_bottom = init_mid - init_height / 2 / action_config['final_axial_compression']
        step2_r = np.sqrt(
            shared_config['N'] * action_config['final_per_particle_volume'] 
            / (step2_top - step2_bottom) / np.pi
        )

        action_config['step1_r'] = step1_r 
        action_config['step1_top'] = step1_top
        action_config['step1_bottom'] = step1_bottom
        
        action_config['step2_r'] = step2_r 
        action_config['step2_top'] = step2_top
        action_config['step2_bottom'] = step2_bottom

        return shared_config_added_data, action_config


    def run_loop(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        if ((self_conf['step1_start'] <= sim.block <= self_conf['step1_end'])
            or 
            (self_conf['step2_start'] <= sim.block <= self_conf['step2_end'])
           ):
            step = 'step1' if (self_conf['step1_start'] <= sim.block <= self_conf['step1_end']) else 'step2'
            ks = [k for k in ['r', 'top', 'bottom']
                  if self_conf[f'{step}_{k}'] is not None]

            cur_vals = {
                k:sim.context.getParameter(f'cylindrical_confinement_{k}')
                for k in ks
            }

            new_vals = {
                k:cur_vals[k] + (
                    (cur_vals[k] - self_conf[f'{step}_{k}']) 
                    * (
                        ((self_conf[f'{step}_end'] + 1 - sim.block - 1) ** self_conf['powerlaw'])
                        / ((self_conf[f'{step}_end'] + 1 - sim.block ) ** self_conf['powerlaw'])
                        - 1
                    )
                )
                for k in ks
            }

            for k in ks:
                sim.context.setParameter(
                    f'cylindrical_confinement_{k}', new_vals[k] * sim.conlen)

            if 'AddTipsTethering' in action_configs:
                if 'top' in ks and 'bottom' in ks:
                    sim.force_dict['Tethers'].setParticleParameters(
                        0, 0, [0, 0, new_vals['bottom']])
                    sim.force_dict['Tethers'].setParticleParameters(
                        1, sim.N-1, [0, 0, new_vals['top']])
                    sim.force_dict['Tethers'].updateParametersInContext(
                        sim.context)


class AddStaticCylinderCompression(SimAction):
    def __init__(
        self,
        k=1.0,
        z_min=None,
        z_max=None,
        r=None,
        per_particle_volume = 1.5*1.5*1.5
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)

    
    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if ((action_config['z_min'] is None) != (action_config['z_max'] is None)):
            raise ValueError('Both z_min and z_max have to be either specified or left as None.')
        coords = shared_config['initial_conformation']
        if (action_config['z_min'] is None):
            action_config['z_min'] = coords[:,2].min()
        elif (action_config['z_min'] == 'bb'):
            action_config['z_min'] = coords[shared_config['backbone']][:,2].min()
        else:
            action_config['z_min'] = action_config['z_min']
            
            
        if (action_config['z_max'] is None):
            action_config['z_max'] = coords[:,2].max()
        elif (action_config['z_max'] == 'bb'):
            action_config['z_max'] = coords[shared_config['backbone']][:,2] .max()
        else:
            action_config['z_max'] = action_config['z_max']


        if ((action_config['r'] is not None) and (action_config['per_particle_volume'] is not None)):
            raise ValueError('Please specify either r or per_particle_volume.')
        elif ((action_config['r'] is None) and (action_config['per_particle_volume'] is None)):
            coords = shared_config['initial_conformation']
            action_config['r'] = ((coords[:,:2]**2).sum(axis=1)**0.5).max()
        elif ((action_config['r'] is None) and (action_config['per_particle_volume'] is not None)):
            action_config['r'] = np.sqrt(
                shared_config['N'] * action_config['per_particle_volume'] 
                / (action_config['z_max'] - action_config['z_min']) / np.pi
            )
        
        return shared_config_added_data, action_config

    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        sim.add_force(
            extra_forces.cylindrical_confinement_2(
                sim_object=sim,
                r=self_conf['r'],
                top=self_conf['z_max'],
                bottom=self_conf['z_min'], 
                k=self_conf['k']
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
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)
    
        n_params = sum([
            i is None 
            for i in [action_config['helix_radius'], 
                      action_config['helix_turn_length'], 
                      action_config['helix_step'], 
                      action_config['axial_compression_factor']]])
        
        if n_params not in [0, 2]:
            raise ValueError('Please specify 0 or 2 out of these four parameters: '
                             'radius, turn_length, step and axis-to-backbone ratio')
                
        if (action_config['helix_radius'] is not None) and (action_config['helix_step'] is not None):
            helix_radius = action_config['helix_radius']
            helix_step = action_config['helix_step']
        elif (action_config['helix_turn_length'] is not None) and (action_config['helix_step'] is not None):
            helix_step = action_config['helix_step']
            helix_radius_squared = ( (action_config['helix_turn_length']) ** 2 
                                      - 
                                      (action_config['helix_step']) ** 2 
                                    ) / np.pi / np.pi / 2.0 / 2.0
            if helix_radius_squared <= 0:
                raise ValueError('The provided values of helix_step and helix_turn_length are incompatible')
            helix_radius = helix_radius_squared ** 0.5
            
        elif (action_config['helix_turn_length'] is not None) and (action_config['helix_radius'] is not None):
            helix_radius = action_config['helix_radius']
            helix_step_squared = ( (action_config['helix_turn_length']) ** 2 
                                      - 
                                      (2 * np.pi * helix_radius) ** 2 )
            if helix_step_squared <= 0:
                raise ValueError('The provided values of helix_step and helix_turn_length are incompatible')
            helix_step = helix_step_squared ** 0.5
            
        elif (action_config['axial_compression_factor'] is not None) and (action_config['helix_radius'] is not None):
            helix_radius = action_config['helix_radius']
            helix_step = 2 * np.pi * helix_radius / np.sqrt(action_config['axial_compression_factor'] ** 2 - 1)
            
        elif (action_config['axial_compression_factor'] is not None) and (action_config['helix_turn_length'] is not None):
            helix_step = action_config['helix_turn_length'] / action_config['axial_compression_factor'] 
            helix_radius_squared = ( (action_config['helix_turn_length']) ** 2 
                                      - 
                                      (helix_step) ** 2 
                                    ) / np.pi / np.pi / 2.0 / 2.0
            if helix_radius_squared <= 0:
                raise ValueError('The provided values of helix_step and helix_turn_length are incompatible')
            helix_radius = helix_radius_squared ** 0.5
        elif (action_config['axial_compression_factor'] is not None) and (action_config['helix_step'] is not None):
            helix_step = action_config['helix_step']
            helix_turn_length = helix_step * action_config['axial_compression_factor']
            helix_radius_squared = ( (helix_turn_length) ** 2 
                                      - 
                                      (helix_step) ** 2 
                                    ) / np.pi / np.pi / 2.0 / 2.0
            if helix_radius_squared <= 0:
                raise ValueError('The provided values of helix_step and helix_turn_length are incompatible')
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        action_config['helix_step'] = helix_step
        action_config['helix_radius'] = helix_radius

        shared_config_added_data['initial_conformation'] = (
            starting_mitotic_conformations.make_helical_loopbrush(
                L=shared_config['N'],
                helix_radius=helix_radius,
                helix_step=helix_step,
                loops=shared_config['loops'],
                random_loop_orientations=action_config['random_loop_orientations']
            )
        )

        return shared_config_added_data, action_config


class GenerateLoopBrushUniformHelixInitialConformation(SimAction):
    def __init__(
        self,
        helix_radius=None,
        helix_step=None,
        axial_compression_factor=None,
        period_particles=None,
        loop_fold='RW',
        loop_layer='all',
        chain_bond_length=1.0,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)
    
        n_params = sum([
            i is not None 
            for i in [action_config['helix_radius'], 
                      action_config['helix_step'], 
                      action_config['axial_compression_factor']]])
        
        if n_params not in [0, 2]:
            raise ValueError('Please specify 0 or 2 out of these four parameters: '
                             'radius, step and axis-to-backbone ratio')
                
        if (action_config['helix_radius'] is not None) and (action_config['helix_step'] is not None):
            helix_radius = action_config['helix_radius']
            helix_step = action_config['helix_step']
        elif (action_config['axial_compression_factor'] is not None) and (action_config['helix_radius'] is not None):
            helix_radius = action_config['helix_radius']
            helix_step = 2 * np.pi * helix_radius / np.sqrt(action_config['axial_compression_factor'] ** 2 - 1)
            
        elif (action_config['axial_compression_factor'] is not None) and (action_config['helix_step'] is not None):
            helix_step = action_config['helix_step']
            helix_turn_length = helix_step * action_config['axial_compression_factor']
            helix_radius_squared = ( 
                helix_turn_length ** 2 - helix_step ** 2 
                ) / np.pi / np.pi / 2.0 / 2.0
            helix_radius = helix_radius_squared ** 0.5
        else:
            helix_radius = 0
            helix_step = int(1e9)

        action_config['helix_step'] = helix_step
        action_config['helix_radius'] = helix_radius

        shared_config_added_data['initial_conformation'] = (
            starting_mitotic_conformations.make_uniform_helical_loopbrush(
                L=shared_config['N'],
                helix_radius=helix_radius,
                helix_step=helix_step,
                period_particles=action_config['period_particles'],
                loops=shared_config['loops'],
                chain_bond_length=action_config['chain_bond_length'],
                loop_fold=action_config['loop_fold'],
            )
        )

        return shared_config_added_data, action_config


class SaveConfiguration(SimAction):
    def __init__(
        self,
        backup = True,
        mode_exists = 'fail', # 'exit' 'overwrite'
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if action_config['mode_exists'] not in ['fail', 'exit', 'overwrite']:
            raise ValueError(f'Unknown mode for saving configuration: {action_config["mode_exists"]}')
        if os.path.exists(shared_config['folder']):
            if action_config['mode_exists'] == 'fail':
                raise OSError(f'The output folder already exists {shared_config["folder"]}')
            elif action_config['mode_exists'] == 'exit':
                sys.exit(0)

        os.makedirs(shared_config['folder'], exist_ok=True)
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


class AddChainsSelectiveRepAttr(SimAction):
    def __init__(
        self,
        chains = ((0, None, 0),),
        bond_length = 1.0,
        wiggle_dist = 0.025,
        stiffness_k = None,
        repulsion_e = 2.5, ## TODO: implement np.inf 
        attraction_e = None,
        attraction_r = None,
        selective_attraction_e = None,
        particle_types = None,
        except_bonds = False,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if hasattr(action_config['chains'], '__iter__') and hasattr(action_config['chains'][0], '__iter__'):
            shared_config_added_data['chains'] = action_config['chains']
        elif hasattr(action_config['chains'], '__iter__') and isinstance(action_config['chains'][0], numbers.Number):
            edges = np.r_[0, np.cumsum(action_config['chains'])]
            chains = [(st, end, False) for st, end in zip(edges[:-1], edges[1:])]
            action_config['chains'] = chains
            shared_config_added_data['chains'] = chains
            
        return shared_config_added_data, action_config


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        nonbonded_force_func = extra_forces.homotypic_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
                        repulsionEnergy=self_conf['repulsion_e'],
                        repulsionRadius=1.0,
                        attractionEnergy=self_conf['attraction_e'],
                        attractionRadius=self_conf['attraction_r'],
                        particleTypes=self_conf['particle_types'],
                        selectiveAttractionEnergy=self_conf['selective_attraction_e'],
                    )
        

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=shared_config['chains'],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength': self_conf['bond_length'],
                    'bondWiggleDistance': self_conf['wiggle_dist'],
                },

                angle_force_func=(
                    None if self_conf['stiffness_k'] is None 
                    else forces.angle_force),
                angle_force_kwargs={
                    'k': self_conf['stiffness_k'] 
                },

                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,

                except_bonds=self_conf['except_bonds']
            )
        )


class AddChainsHeteropolymerRepAttr(SimAction):
    def __init__(
        self,
        chains = ((0, None, 0),),
        bond_length = 1.0,
        wiggle_dist = 0.025,
        stiffness_k = None,
        repulsion_e = 2.5, ## TODO: implement np.inf 
        attraction_e = None,
        attraction_r = None,
        particle_types = None,
        except_bonds = False,
    ):
        params = {k:v for k,v in locals().items() if k not in ['self']} # This line must be the first in the function.
        params['chains'] = np.array(chains, dtype=np.object)
        super().__init__(**params)


    def configure(self, shared_config, action_configs):
        shared_config_added_data, action_config = super().configure(
            shared_config, action_configs)

        if hasattr(action_config['chains'], '__iter__') and hasattr(action_config['chains'][0], '__iter__'):
            shared_config_added_data['chains'] = action_config['chains']
        elif hasattr(action_config['chains'], '__iter__') and isinstance(action_config['chains'][0], numbers.Number):
            edges = np.r_[0, np.cumsum(action_config['chains'])]
            chains = np.array(
                [(st, end, False) for st, end in zip(edges[:-1], edges[1:])],
                dtype=np.object)
            action_config['chains'] = chains
            shared_config_added_data['chains'] = chains
            
        return shared_config_added_data, action_config


    def run_init(self, shared_config, action_configs, sim):
        # do not use self.params!
        # only use parameters from action_configs[self.name] and shared_config
        self_conf = action_configs[self.name]

        nonbonded_force_func = extra_forces.heteropolymer_quartic_repulsive_attractive
        nonbonded_force_kwargs = dict(
                        repulsionEnergy=self_conf['repulsion_e'],
                        repulsionRadius=1.0,
                        attractionEnergies=self_conf['attraction_e'],
                        attractionRadius=self_conf['attraction_r'],
                        particleTypes=(
                            shared_config['particle_types']
                            if self_conf['particle_types'] is None 
                            else self_conf['particle_types']
                        )
        )
    
        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=shared_config['chains'],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength': self_conf['bond_length'],
                    'bondWiggleDistance': self_conf['wiggle_dist'],
                },

                angle_force_func=(
                    None if self_conf['stiffness_k'] is None 
                    else forces.angle_force),
                angle_force_kwargs={
                    'k': self_conf['stiffness_k'] 
                },

                nonbonded_force_func=nonbonded_force_func,
                nonbonded_force_kwargs=nonbonded_force_kwargs,

                except_bonds=self_conf['except_bonds']
            )
        )


