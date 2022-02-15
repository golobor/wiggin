import collections
from polychrom.forces import openmm, _check_bonds


def adjustable_harmonic_bonds(
    sim_object,
    bonds,
    bondWiggleDistance=0.05,
    bondLength=1.0,
    name="adjustable_harmonic_bonds",
    override_checks=False,
):
    """
    
    Constant force bond force. Energy is roughly linear with estension 
    after r=quadraticPart; before it is quadratic to make sure the force
    is differentiable. 
    
    Force is parametrized using the same approach as bond force:
    it reaches U=kT at extension = bondWiggleDistance 
    
    Note that, just as with bondForce, mean squared extension 
    is actually larger than wiggleDistance by sqrt(2) factor. 
    
    Parameters
    ----------
    
    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float
        Displacement at which bond energy equals 1 kT. 
        Can be provided per-particle.
    bondLength : float
        The length of the bond.
        Can be provided per-particle.
    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated bonds
    if not override_checks:
        _check_bonds(bonds, sim_object.N)

    energy = (
        "kT * ((r / l_unit - r0) / wiggle_dist)^2"
    )
    force = openmm.CustomBondForce(energy)
    force.name = name

    force.addGlobalParameter("l_unit", sim_object.conlen)
    force.addGlobalParameter("kT", sim_object.kT)

    bond_params = []

    if isinstance(bondWiggleDistance, collections.abc.Iterable):
        force.addPerBondParameter("wiggle_dist")
        bond_params.append(bondWiggleDistance)
    else:
        force.addGlobalParameter("wiggle_dist", bondWiggleDistance)

    if isinstance(bondLength, collections.abc.Iterable):
        force.addPerBondParameter("r0")
        bond_params.append(bondLength)
    else:
        force.addGlobalParameter("r0", bondLength)

    if bond_params:
        bond_params = list(zip(*bond_params))

    for bondIdx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError(
                "\nCannot add bond with monomers %d,%d that"
                "are beyound the polymer length %d" % (i, j, sim_object.N)
            )

        force.addBond(
            int(i),
            int(j),
            bond_params[bondIdx] if bond_params else []
        )

    return force


def quartic_repulsive(sim_object, trunc=3.0, radiusMult=1.0, name="quartic_repulsive"):
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

    force.addGlobalParameter("REPe", trunc * sim_object.kT)
    force.addGlobalParameter("REPsigma", radius)

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

