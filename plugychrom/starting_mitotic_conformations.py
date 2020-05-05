import looplib.looptools
import numpy as np


def norm(vector):
    return np.sqrt(np.dot(vector, vector))


# def swap_nearby_particles(d, portion, cutoff=1.5, separation_cutoff=1):
#     newd = np.copy(d)
#     contacts = polymerScalings.giveContacts(d, cutoff)
#     numSwapped = 0
#     for i,j in contacts:
#         if abs(i-j)>separation_cutoff and np.random.random() < portion:
#             newd[i], newd[j] = d[j], d[i]
#             numSwapped += 1
#     print('{0} particles swapped...'.format(numSwapped))

#     return newd


def make_catenated_pair(core_conformation, linking_number, radius, shorten_conformation=True):
    if core_conformation.shape[0] == 3:
        core_conformation = core_conformation.T

    N = core_conformation.shape[0]
    rotation_step = np.pi * linking_number / N
    bond_length = np.sqrt(
        ((core_conformation[:-1] - core_conformation[1:]) ** 2).sum(
            axis=1)).mean()

    tangents = core_conformation[1:,:] - core_conformation[:-1,:]
    tangents = np.vstack([tangents, tangents[-1]])

    tangents /= np.sqrt((tangents * tangents).sum(axis=1))[np.newaxis].T

    def conformation_float_position(conformation, idx):
        return (conformation[int(np.floor(idx))] * (idx - np.floor(idx)) +
                conformation[int(np.ceil(idx))] * (np.floor(idx+1.0) - idx))

    traj1 = np.zeros(shape=core_conformation.shape)
    traj2 = np.zeros(shape=core_conformation.shape)

    if shorten_conformation:
        core_shift = np.sqrt(
            bond_length ** 2.0 - 2.0 * radius * radius * (1.0 - np.cos(rotation_step)))
    else:
        core_shift = bond_length

    shifts = np.zeros(shape=core_conformation.shape)
    shifts[0] = np.cross(tangents[0], [0,0,1.0])
    shifts[0] *= radius / norm(shifts[0])

    core_positions = np.zeros(shape=core_conformation.shape)
    core_positions[0] = core_conformation[0]

    for i in range(1, N):

        core_positions[i] = conformation_float_position(
            core_conformation, i * core_shift / bond_length)

        local_tangent = conformation_float_position(tangents, i * core_shift / bond_length)
        local_tangent /= norm(local_tangent)

        prev_shift = np.copy(shifts[i-1])
        prev_shift -= local_tangent * np.dot(local_tangent, prev_shift)
        prev_shift /= norm(prev_shift)

        orth_shift = np.cross(prev_shift, local_tangent)

        new_shift = (local_tangent * core_shift
                     + prev_shift * np.cos(rotation_step)
                     + orth_shift * np.sin(rotation_step))
        new_shift *= radius / norm(new_shift)

        shifts[i] = new_shift

    return (core_positions + shifts), (core_positions - shifts)


def bended_bb(bb_len, angle_degrees, bend_region):
    bb_traj = np.zeros(shape=(bb_len,3))
    u_0 = np.array([1.0,0,np.tan((90.0-angle_degrees/2.0)/180.0*np.pi)])
    u_0 /= norm(u_0)
    u_f = np.array([1.0,0,-np.tan((90.0-angle_degrees/2.0)/180.0*np.pi)])
    u_f /= norm(u_f)
    for i in range(1,bb_len):
        if i < bend_region[0]:
            u = u_0
        elif bend_region[0] <= i <  bend_region[1]:
            u = (bend_region[1]-i) * u_0 + (i-bend_region[0]) * u_f
            u /= norm(u)
        else:
            u = u_f
        bb_traj[i] = bb_traj[i-1] + u
    return bb_traj


def fold_loopbrush_backbone(L, loops, bb_traj, loop_plane_normal=None):
    coords = np.zeros(shape=(L,3))
    loopstarts = np.array([min(i) for i in loops])
    loopends = np.array([max(i) for i in loops])
    looplens = loopends - loopstarts

    bbidxs = np.array(
        list(range(0,loopstarts[0]+1))
        + sum([list(range(loopends[i],loopstarts[i+1]+1))
               for i in range(len(loops)-1)],
              [])+
        list(range(loopends[-1], L)))
    coords[bbidxs] = bb_traj[:len(bbidxs)]

    for i in range(len(loops)):
        if loop_plane_normal is None:
            bb_u = coords[loops[i][1]] - coords[loops[i][0]]
        else:
            bb_u = loop_plane_normal
        u = np.cross(bb_u, bb_u+(np.random.random(3)*0.2-0.1))
        u /= (u**2).sum()**0.5
        for j in range(looplens[i] // 2):
            coords[loopstarts[i]+j+1] = coords[loopstarts[i]+j] + u
            coords[loopends[i]-j-1]   = coords[loopends[i]-j]   + u
    return coords


def make_folded_loopbrush(L, step, bb_perlength, loops,
                          polar_range=(0,1), go_vertical=False):

    loopstarts = np.array([min(i) for i in loops])
    loopends = np.array([max(i) for i in loops])

    bbidxs = np.array(
        list(range(0,loopstarts[0]+1))
        + sum([list(range(loopends[i],loopstarts[i+1]+1))
               for i in range(len(loops)-1)],
              [])+
        list(range(loopends[-1], L)))

    bb_traj = np.zeros(shape=(len(bbidxs),3))
    u = np.array([0.0,0.0,1.0])
    u0 = np.copy(u)
    for i in range(1,len(bbidxs)):
        if i % bb_perlength == 0:
            prev_u = u0 if go_vertical else u
            orth_u = np.cross(prev_u, prev_u+(np.random.random(3)*0.2-0.1))
            orth_u /= (orth_u**2).sum()**0.5
            polar = (polar_range[1] - polar_range[0]) * np.random.random() + polar_range[0]
            u = polar * prev_u + np.sqrt(1.0 - polar * polar) * orth_u

        bb_traj[i] = bb_traj[i-1] + u

    return fold_loopbrush_backbone(
        L, loops, bb_traj, np.array([0,0,1]) if go_vertical else None)


def make_spiral(L, radius, step, linear_density=1.0):
    phase_step = 1.0 / linear_density / np.sqrt(
        radius * radius + step*step/4.0/np.pi/np.pi)

    coords = np.zeros(shape=(L,3))
    coords[:,0] = radius * np.sin(np.arange(L) * phase_step)
    coords[:,1] = radius * np.cos(np.arange(L) * phase_step)
    coords[:,2] = phase_step / 2.0 / np.pi * step * np.arange(L)

    return coords


#def biased_RW(L, step, avg_len):
#    if L * step < avg_len:
#        raise Exception("Not possible, sir!")
#
#    bias = (avg_len / float(L) / float(step)) / 2.0
#    steps = step * (np.random.random(L-1) < 0.5 + bias)
#
#    xs = np.cumsum(np.r_[0, steps])
#    return xs
#

def biased_RW(L, step, avg_len):                            
    if L * step < avg_len:                                  
        raise Exception("Not possible, madam/sir!")               
                                                            
    bias = (avg_len / float(L) / float(step)) / 2.0         
    steps = step * (2*(np.random.random(L-1) < 0.5 + bias) -1)
    xs = np.cumsum(np.r_[0, steps])                         
    return xs                                               


def brownian_bridge(N, step, final_len):                            
    if N * step < final_len:
        raise Exception("Not possible, madam/sir!")               
    
    xs = [0]
    for i in range(1,N):
        bias = ( 1.0 - (final_len - xs[-1]) /step/ (N-i)) / 2.0
        xs.append(xs[-1] + (-step if np.random.random()<bias else step))
    return np.array(xs)


def make_helical_loopbrush(
        L,
        helix_radius,
        helix_step,
        loops,
        bb_linear_density=1.0,
        random_loop_orientations=False,
        bb_random_shift=0):
    '''
    Generate a conformation of a loop brush with a helically folded backbone.
    In this conformation, loops are folded in half and project radially
    from the backbone. 
    
    Parameters
    ----------
    L : int
        Number of particles.
    helix_radius: float
        Radius of the helical backbone.
    helix_step: float
        Axial step of the helical backbone.
    loops: a list of tuples [(int, int)]
        Particle indices of (start, end) of each loop.
    bb_linear_density: float
        The linear density of the backbone, 
        num_particles / unit of backbone length 
    random_loop_orientations: bool
        If True, then align loops at random angles, 
        otherwise align them along the radius set by
        the location of their base with respect to the 
        center of the helix.
    bb_random_shift : float
        Add a random shift along all three coordinates to the backbone.
        The default value is 0.
    Returns 
    -------
    coords: np.ndarray
        An Lx3 array of particle coordinates.
    
    '''
    coords = np.zeros(shape=(L,3))
    root_loops = loops[looplib.looptools.get_roots(loops)]
    loopstarts = np.array([min(i) for i in root_loops])
    loopends = np.array([max(i) for i in root_loops])
    looplens = loopends - loopstarts

    if len(root_loops)>0:
        bbidxs = np.concatenate(
            [np.arange(0,loopstarts[0]+1)]
            + [np.arange(loopends[i],loopstarts[i+1]+1)
               for i in range(len(root_loops)-1)]
            + [np.arange(loopends[-1], L)])
    else:
        bbidxs = range(L)
    bb_len = len(bbidxs)

    helix_turn_len = np.sqrt(
        (2.0 * np.pi * helix_radius)**2 + helix_step**2)
    helix_total_winding = 2.0 * np.pi * (bb_len - 1) / bb_linear_density / helix_turn_len

    bb_phases = np.linspace(0, helix_total_winding, bb_len)

    coords[bbidxs] = np.vstack(
        [helix_radius * np.sin(bb_phases),
         helix_radius * np.cos(bb_phases),
         bb_phases / 2.0 / np.pi * helix_step]).T
    coords[bbidxs] += (
        np.random.random(bb_len * 3).reshape(bb_len, 3) * bb_random_shift)

    for i in range(len(root_loops)):
        if random_loop_orientations:
            bb_u = coords[root_loops[i][1]] - coords[root_loops[i][0]]
            u = np.cross(bb_u, bb_u+(np.random.random(3)*0.2-0.1))
            u[2] = 0
        else:
            u = (coords[root_loops[i][0]] + coords[root_loops[i][1]])/2
            u[2] = 0

        u /= (u**2).sum()**0.5
        for j in range(looplens[i] // 2):
            coords[loopstarts[i]+j+1] = coords[loopstarts[i]+j] + u
            coords[loopends[i]-j-1]   = coords[loopends[i]-j]   + u

    return coords


def make_pseudo_globule(step_size, N, segment_length=1,
                        vonmisesmju=0,
                        vonmiseskappa=0):
    u = np.zeros(N // segment_length + 1)
    i = 0
    '''ugly, but there is not vonMises-Fisher (on a 3d sphere) in Python'''
    while True:
        theta = np.random.vonmises(vonmisesmju, vonmiseskappa)
        if np.random.random() < np.sin(theta):
            u[i] = np.cos(theta)
            i += 1
            if i >= u.size:
                break
    u = -u

    phi = 2.0 * np.pi * np.random.uniform(0., 1., N // segment_length + 1)
    drs = np.vstack([
        np.sqrt(1. - u * u) * np.cos(phi),
        np.sqrt(1. - u * u) * np.sin(phi),
        u]).T
    drs = iter(drs)
    r = np.zeros(shape=(N,3))
    r[0] = [0,1,0]

    i = 1
    while True:
        norm_r = r[i-1] / (((r[i-1]**2).sum())**0.5)
        norm_ort1 = np.cross(norm_r, [1,0,0])
        norm_ort1 /= (((norm_ort1**2).sum())**0.5)
        norm_ort2 = np.cross(norm_r, norm_ort1)
        norm_ort2 /= (((norm_ort2**2).sum())**0.5)
        dr = next(drs)
        dr_rotated = norm_r * dr[2] + norm_ort1 * dr[0] + norm_ort2 * dr[1]

        for j in range(i, min(N,i+segment_length)):
            r[j] = r[j-1] + dr_rotated
        i += segment_length
        if i >= N:
            break

    return r


