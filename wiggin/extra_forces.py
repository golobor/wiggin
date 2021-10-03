import collections
import logging

import numpy as np

from polychrom.forces import openmm


def quartic_repulsive(
    sim_object, trunc=3.0, radiusMult=1.0, name="quartic_repulsive"
):
    """
    This is one of the simplest repulsive potential, a polynomial of fourth power.
    It has the value of `trunc` at zero.
    It seems to be ~10% slower than polynomial_repulsive, but more stable.
    
    Parameters
    ----------
    trunc : float
        the energy value around r=0
    """
    
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    repul_energy = (
        "(1-2*rnorm2+rnorm2*rnorm2) * REPe;"
        "rnorm2 = rnorm*rnorm;"
        "rnorm = r/REPsigma"
    )

    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name

    force.addGlobalParameter("REPe", trunc * sim_object.kT )
    force.addGlobalParameter("REPsigma", radius )
    
    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)

    return force


def quartic_repulsive_attractive(
    sim_object,
    repulsionEnergy=3.0,
    repulsionRadius=1.0,
    attractionEnergy=0.5,
    attractionRadius=2.0,
    name="quartic_repulsive_attractive",
):
    """
    This is one of the simplest potentials that combine a soft repulsive core with 
    an attractive shell. It is based on 4th-power polynomials.
     
    Parameters
    ----------
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the radius of the attractive part of the potential.
        E(`attractionRadius`) = 0,
        E'(`attractionRadius`) = 0
    """

    nbCutOffDist = sim_object.conlen * attractionRadius

    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep =(1-2*rnorm2+rnorm2*rnorm2) * REPe;"
        "rnorm2 = rnorm*rnorm;"
        "rnorm = r/REPsigma;"
        ""
        "Eattr = (-1)* (1-2*rnorm_shift2+rnorm_shift2*rnorm_shift2) * ATTRe;"
        "rnorm_shift2 = rnorm_shift*rnorm_shift;"
        "rnorm_shift = (r - REPsigma - ATTRdelta)/ATTRdelta"
    )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter(
        "ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0
    )

    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)

    return force


def homotypic_quartic_repulsive_attractive(
    sim_object,
    particleTypes,
    repulsionEnergy=3.0,
    repulsionRadius=1.0,
    attractionEnergy=3.0,
    attractionRadius=1.5,
    selectiveAttractionEnergy=1.0,
    name="homotypic_quartic_repulsive_attractive",
):
    """
    This is one of the simplest potentials that combine a soft repulsive core with 
    an attractive shell. It is based on 4th-power polynomials.
    
    Monomers of type 0 do not get extra attractive energy.

     
    Parameters
    ----------
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the radius of the attractive part of the potential.
        E(`attractionRadius`) = 0,
        E'(`attractionRadius`) = 0
    """

    nbCutOffDist = sim_object.conlen * attractionRadius

    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep =(1-2*rnorm2+rnorm2*rnorm2) * REPe;"
        "rnorm2 = rnorm*rnorm;"
        "rnorm = r/REPsigma;"
        ""
        "Eattr = (-1)* (1-2*rnorm_shift2+rnorm_shift2*rnorm_shift2) * ATTReTot;"
        "ATTReTot = ATTRe + delta(type1-type2) * (1-delta(type1)) * (1-delta(type2)) * ATTReAdd;"
        "rnorm_shift2 = rnorm_shift*rnorm_shift;"
        "rnorm_shift = (r - REPsigma - ATTRdelta)/ATTRdelta"
    )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter(
        "ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0
    )
    force.addGlobalParameter("ATTReAdd", selectiveAttractionEnergy * sim_object.kT)
    
    force.addPerParticleParameter("type")

    for i in range(sim_object.N):
        force.addParticle((float(particleTypes[i]),))

    force.setCutoffDistance(nbCutOffDist)

    return force


def max_dist_bonds(
        sim_object,
        bonds,
        max_dist=1.0,
        k=5,
        axes=['x', 'y', 'z'],
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
    force.addGlobalParameter("k", k / sim_object.conlen)
    force.addGlobalParameter("t",  0.1 / k * sim_object.conlen)
    force.addGlobalParameter("tt", 0.01 * sim_object.conlen)
    force.addGlobalParameter("max_dist", max_dist * sim_object.conlen)
    
    for _, (i, j) in enumerate(bonds):
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
        force.addGlobalParameter("t", (1. / k.mean()) * sim_object.conlen / 10.)
    else:
        force.addGlobalParameter("t", sim_object.conlen)
    force.addPerParticleParameter("kx")
    force.addPerParticleParameter("ky")
    force.addPerParticleParameter("kz")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    if positions == "current":
        positions = [sim_object.data[i] for i in particles]
    else:
        positions = np.array(positions) * sim_object.conlen

    for i, (kx,ky,kz), (x,y,z) in zip(particles, k, positions):  # adding all the particles on which force acts
        i = int(i)
        force.addParticle(i, (kx * sim_object.kT / sim_object.conlen,
                              ky * sim_object.kT / sim_object.conlen,
                              kz * sim_object.kT / sim_object.conlen,
                              x,y,z
                             )
                         )
        if sim_object.verbose == True:
            print("particle %d tethered! " % i)
    
    return force


def angular_tether_particles(
        sim_object, 
        particles=None, 
        angle_wiggle=np.pi / 16,
        min_r=0.1,
        angles="current",
        name="linear_tethers"
        ):
    """tethers the angles of particles in the xy plane.
    
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
        "k * (1 - (x * x0 + y * y0) / sqrt(x*x + y*y + t*t) / sqrt(x0*x0 + y0*y0 ) )"
    )

    force = openmm.CustomExternalForce(energy)
    force.name = name

    if particles is None:
        particles = range(sim_object.N)
        N_tethers = sim_object.N
    else:
        particles = [sim_object.N+i if i < 0 else i for i in particles]
        N_tethers = len(particles)

    k = 1 / angle_wiggle / angle_wiggle
    
    force.addGlobalParameter("t", min_r * sim_object.conlen)
    force.addGlobalParameter("k", k * sim_object.kT)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")

    if angles == "current":
        angles = np.array([sim_object.data[i] for i in particles])[:, :2]
    else:
        angles = np.array(angles)
        if angles.ndim == 1:
            angles = np.vstack([
                np.cos(angles), np.sin(angles)
            ]).T * sim_object.conlen
        elif (angles.ndim == 2) and (angles.shape[0] == N_tethers) and (angles.shape[1] == 2):
            angles = np.array(angles) * sim_object.conlen
        else:
            raise ValueError('Unknown format for angles')

    for i, (x0, y0) in zip(particles, angles):  # adding all the particles on which force acts
        force.addParticle(int(i), (float(x0), float(y0)))
        if sim_object.verbose:
            logging.debug("Particle angle %d tethered! " % i)
            
    return force




def heteropolymer_quartic_repulsive_attractive(
    sim_object,
    particleTypes,
    attractionEnergies,
    repulsionEnergy=3.0,  # base repulsion energy for **all** particles
    repulsionRadius=1.0,
    attractionRadius=1.5,
    keepVanishingInteractions=False,
    name="heteropolymer_quartic_repulsive_attractive",
):
    """
    A version of smooth square well potential that enables the simulation of
    heteropolymers. Every monomer is assigned a number determining its type,
    then one can specify additional attraction between the types with the
    interactionMatrix. Repulsion between all monomers is the same, except for 
    extraHardParticles, which, if specified, have higher repulsion energy. 
    
    The overall potential is the same as in :py:func:`polychrom.forces.smooth_square_well`
    
    Treatment of extraHard particles is the same as in :py:func:`polychrom.forces.selective_SSW`

    This is an extension of SSW (smooth square well) force in which:
    
    a) You can give monomerTypes (e.g. 0, 1, 2 for A, B, C)
       and interaction strengths between these types. The corresponding entry in
       interactionMatrix is multiplied by selectiveAttractionEnergy to give the actual
       **additional** depth of the potential well.        
    b) You can select a subset of particles and make them "extra hard". See selective_SSW force for descrition. 
    
    Force summary
    *************
    
    Potential is the same as smooth square well, with the following parameters for particles i and j: 
    
    * Attraction energy (i,j) = attractionEnergy + selectiveAttractionEnergy * interactionMatrix[i,j] 

    * Repulsion Energy (i,j) = repulsionEnergy + selectiveRepulsionEnergy;  if (i) or (j) are extraHard
    * Repulsion Energy (i,j) = repulsionEnergy;  otherwise 

    Parameters
    ----------

    interactionMatrix: np.array
        the **EXTRA** interaction strenghts between the different types.
        Only upper triangular values are used. See "Force summary" above 
    monomerTypes: list of int or np.array
        the type of each monomer, starting at 0
    extraHardParticlesIdxs : list of int
        the list of indices of the "extra hard" particles. The extra hard
        particles repel all other particles with extra
        `selectiveRepulsionEnergy`
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the maximal range of the attractive part of the potential.
    selectiveRepulsionEnergy: float
        the **EXTRA** repulsion energy applied to the "extra hard" particles
    selectiveAttractionEnergy: float
        the **EXTRA** attraction energy (prefactor for the interactionMatrix interactions)
    keepVanishingInteractions : bool
        a flag that determines whether the terms that have zero interaction are
        still added to the force. This can be useful when changing the force
        dynamically (i.e. switching interactions on at some point)
    """

    if not attractionEnergies:
        raise ValueError('Please provide interaction energies as a list of (i,j, energy)!')
    
    n_interactions = len(set((i,j) for i,j,e in attractionEnergies))
    n_interactions_simplified = len(set((min(i,j), max(i,j)) for i,j,e in attractionEnergies)) 
    if (n_interactions != n_interactions_simplified):
        raise ValueError('Each pairwise interaction should be specified only once!')
    
    attractionEnergiesSym = (
        list(attractionEnergies) 
        + [(j,i,e) for (i,j,e) in attractionEnergies if i != j]
    )

    # Check type info for consistency
    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep =(1-2*rnorm2+rnorm2*rnorm2) * REPe;"
        "rnorm2 = rnorm*rnorm;"
        "rnorm = r/REPsigma;"
        ""
        "Eattr = (-1)* (1-2*rnorm_shift2+rnorm_shift2*rnorm_shift2) * ATTReTot * kT;"   
    )
    
    energy += (
        'ATTReTot = (' 
        +' + '.join(
            [f"delta(type1-{i})*delta(type2-{j})*INT_{i}_{j}"
             for i, j, e in attractionEnergiesSym])
        + ");" 
    )

    energy += (
        "rnorm_shift2 = rnorm_shift*rnorm_shift;"
        "rnorm_shift = (r - REPsigma - ATTRdelta)/ATTRdelta"
    )


    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("kT", sim_object.kT)
    force.addGlobalParameter(
        "ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0
    )

    for i, j, e in attractionEnergiesSym:
        force.addGlobalParameter(f"INT_{i}_{j}", e)

    force.addPerParticleParameter("type")

    for i in range(sim_object.N):
        force.addParticle((float(particleTypes[i]),))

    return force




def cylindrical_confinement_2(
    sim_object, 
    r,  
    k=1.0,
    transition_width=3, 
    top=None,
    bottom=None,
    name="cylindrical_confinement"
):

    if (bottom is not None and top is not None):
        force = openmm.CustomExternalForce(
            "kT * k * ("
            "   step(dr)  * dr  * dr * dr  / (dr * dr + t*t)"
            " + step(dZb) * dZb * dZb * dZb / (dZb * dZb + t*t)"
            " + step(dZt) * dZt * dZt * dZt / (dZt * dZt + t*t)"
            ");"
            "dr = sqrt(x^2 + y^2) - r;"
            "dZt = z - top;"
            "dZb = bottom - z;"
        )
        force.addGlobalParameter("top", top * sim_object.conlen)
        force.addGlobalParameter("bottom", bottom * sim_object.conlen)
    else:
        force = openmm.CustomExternalForce(
            "kT * k * step(dr) * dr * dr * dr/(dr*dt+t*t);"
            "dr = sqrt(x^2 + y^2) - r"
        )
    force.name = name

    for i in range(sim_object.N):
        force.addParticle(i, [])

    force.addGlobalParameter("k", k /  sim_object.conlen)
    force.addGlobalParameter("kT", sim_object.kT)
    force.addGlobalParameter("r", r * sim_object.conlen)
    force.addGlobalParameter("t", transition_width * sim_object.conlen)
    
    return force

