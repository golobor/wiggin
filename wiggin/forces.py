from polychrom.forces import openmm


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

