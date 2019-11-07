import os, shelve, itertools, logging, collections

import numpy as np

from polychrom import simulation, forces, forcekits
from polychrom.forces import openmm, nm


from .simconstructor import AttrDict, SimulationAction

from . import starting_mitotic_conformations




logging.basicConfig(level=logging.INFO)


def max_dist_bonds(
        sim_object,
        bonds,
        max_dist=1.0,
        k = 5,
        axes=['x','y','z'],
        name="max_dist_bonds",
        ):
    """Adds harmonic bonds
    Parameters
    ----------
    
    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float
        Average displacement from the equilibrium bond distance.
        Can be provided per-particle.
    bondLength : float
        The length of the bond.
        Can be provided per-particle.
    """
    
    r_sqr_expr = '+'.join([f'({axis}1-{axis}2)^2' for axis in axes])
    energy = ("kt * k * step(dr) * (sqrt(dr*dr + t*t) - t);"
            + "dr = sqrt(r_sqr + tt^2) - max_dist + 10*t;"
            + 'r_sqr = ' + r_sqr_expr
    )

    print(energy)

    force = openmm.CustomCompoundBondForce(2, energy)
    force.name = name

    force.addGlobalParameter("kt", sim_object.kT)
    force.addGlobalParameter("k", k / nm)
    force.addGlobalParameter("t",  0.1 / k * nm)
    force.addGlobalParameter("tt", 0.01 * nm)
    force.addGlobalParameter("max_dist", max_dist * nm)
    
    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError(
                "\nCannot add bond with monomers %d,%d that"\
                "are beyound the polymer length %d" % (i, j, sim_object.N))
        
        force.addBond((int(i), int(j)), []) 

    return force


def linear_tether_particles(
        sim_object, 
        particles=None, 
        k=5, 
        positions="current",
        name="linear_tethers"
        ):
    """tethers particles in the 'particles' array.
    Increase k to tether them stronger, but watch the system!

    Parameters
    ----------

    particles : list of ints
        List of particles to be tethered (fixed in space).
        Negative values are allowed. If None then tether all particles.
    k : int, optional
        The steepness of the tethering potential.
        Values >30 will require decreasing potential, but will make tethering 
        rock solid.
        Can be provided as a vector [kx, ky, kz].
    """
    
    energy = (
        "   kx * ( sqrt((x - x0)^2 + t*t) - t ) "
        " + ky * ( sqrt((y - y0)^2 + t*t) - t ) "
        " + kz * ( sqrt((z - z0)^2 + t*t) - t ) "
    )

    force = openmm.CustomExternalForce(energy)
    force.name = name

    if particles is None:
        particles = range(sim_object.N)
        N_tethers = sim_object.N
    else:
        particles = [sim_object.N+i if i<0 else i 
                    for i in particles]
        N_tethers = len(particles)


    if isinstance(k, collections.abc.Iterable):
        k = np.array(k)
        if k.ndim == 1:
            if k.shape[0] != 3:
                raise ValueError('k must either be either a scalar, a vector of 3 elements or an (Nx3) matrix!')
            k = np.broadcast_to(k, (N_tethers,3))
        elif k.ndim == 2:
            if (k.shape[0] != N_tethers) and (k.shape[1] != 3):
                raise ValueError('k must either be either a scalar, a vector of 3 elements or an (Nx3) matrix!')
    else:
        k = np.broadcast_to(k, (N_tethers,3))

    if k.mean():
        force.addGlobalParameter("t", (1. / k.mean()) * nm / 10.)
    else:
        force.addGlobalParameter("t", nm)
    force.addPerParticleParameter("kx")
    force.addPerParticleParameter("ky")
    force.addPerParticleParameter("kz")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    if positions == "current":
        positions = [sim_object.data[i] for i in particles]
    else:
        positions = sim_object.addUnits(positions)

    for i, (kx,ky,kz), (x,y,z) in zip(particles, k, positions):  # adding all the particles on which force acts
        i = int(i)
        force.addParticle(i, (kx * sim_object.kT / nm,
                              ky * sim_object.kT / nm,
                              kz * sim_object.kT / nm,
                              x,y,z
                             )
                         )
        if sim_object.verbose == True:
            print("particle %d tethered! " % i)
    
    return force





class GenerateSingleLayerLoops(SimulationAction):
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
        loops = looplib.random_loop_arrays.exponential_loop_array(
                N, 
                action_config['loop_size'],
                action_config['loop_spacing']
        )

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


class GenerateTwoLayerLoops(SimulationAction):
    _default_params = AttrDict(
        inner_loop_size = 200,
        outer_loop_size = 200 * 4,
        inner_loop_spacing = 1,
        outer_loop_spacing = 1,
        outer_inner_offset= 1,
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
            shared_config_added_data.get('loops',[])
            + loops
        )

        action_config['inner_loops'] = inner_loops
        action_config['outer_loops'] = outer_loops

        shared_config_added_data['backbone'] = looplib.looptools.get_backbone(
                outer_loops, N=N)

        return shared_config_added_data, action_config


class AddLoops(SimulationAction):
    _default_params = AttrDict(
        wiggle_dist=0.05,
        bond_length=1.0
    )

    def run_init(self, shared_config, action_configs, sim):
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

    def run_init(self, shared_config, action_configs, sim):
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
    _default_params = AttrDict(
        k=[0,0,5],
        particles=[0, -1],
        positions='current',
    )


    def run_init(self, shared_config, action_configs, sim):
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
    _default_params = AttrDict(
        final_per_particle_volume = 1.5*1.5*1.5,
        final_axial_compression = 1,
        powerlaw = 2.0,
        initial_block = 1,
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


    def run_loop(self, shared_config, action_configs, sim):
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


