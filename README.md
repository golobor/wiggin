# wiggin

## Introduction
wiggin is an experimental high-level modular framework for building polymer simulations.

wiggin's key idea is that any simulation can be decomposed into as a series of atomistic "actions". Such actions can declare or modify some aspect of the system or perform any general task or computation. E.g., an action can initialize a simulation object, generate random positions of cis-loops, add a force that forms chains or imposes an external constraint, perform simulation steps, etc. **wiggin enables building of simulations from arbiratrary reusable blocks, while maintaining readability of the resuling code**.

## Technical notes

### Execution order:
0. **Create a SimConstructor object.** SimConstructor builds a simulation from individual code blocks aka "actions".
1. **Define and add individual actions.**
2. **Configure actions.** Many elements of polymer simulations involve preliminary computations outside of the polymer domain, e.g. generation of a random initial conformation or cis-loop positions (when simulating mitotic chromosomes). Such computations have to be performed only once and should not be re-executed when simulation is continued after a break. These kind of computations are performed at the *configuration* step. 
3. **Initialize the simulation.** All polymer simulations must also be *initialized*. At this stage, we need to create a polychrom Simulation object, add various forces, set the initial coordinates and optimize these conformations. 
4. **Run the simulation loop.** Finally, all simulations involve a computation loop, i.e. a repeated computation of simulations steps, that is periodically interrupted to save the intermediate results, perform checks and updates. 


## Q&A
- **Should I use wiggin instead of polychrom?** The main goal of wiggin is not to replace polychrom, but rather to organize complex polychrom simulations and to enable combinatorial experimentation with such simulations. Thus, users of wiggin must already be familiar with polychrom. 

- **How do I create a new action?**. An new action must define up to 6 elements:
    1. Arguments. These are provided as a field of a `dataclass`<https://docs.python.org/3/library/dataclasses.html>.
    2. Entries of the shared config that will be read and written by the action.
    3. configure()
    4. spawn_actions() - optional 
    5. run_init() - optional
    6. run_loop() - optional

- **Why does wiggin use python dataclasses?** Although rarely used, dataclasses provide a concise and readable syntax to define new Action classes. With dataclasses, we minimize the amount of boilerplate code needed to define a new Action.

- **In which order are actions configured and executed?** By default, actions are configured and executed in the order of their addition. One can customize the execution order using extra arguments of SimConstructor.add_action(), but we would recommend avoiding it.

- **What is shared config and how can I use it?** Shared config of a SimConstructor is a dictionary that can be accessed and modified by actions. All entries read and written by an Action *must* be declared in class variables `_reads_shared` and `_writes_shared`. By default, the following keys of the shared config are reserved:
    - name - name of the simulation. Can be generated from the non-default parameters using auto_name()
    - N - the number of particles in the simulation
    - initial_conformation - the initial coordinates of all particles in the system
    - folder - the folder to store the configuration and the output of the simulation

- **How can I enable interactions between my actions?** There are two way to create dependent actions: 
    - use shared config. Action can communicate by reading and writing entries of the shared config of their SimConstructor. Use this pattern when the resulting entries are generalizeable enough to potentially be used by other actions (i.e. chains, cis-loops, particle types, etc.)
    - create meta-actions that spawn (and pass configured data into) dependent actions. Use this pattern for custom and complex interactions. 

- **What is the preferred naming pattern for actions?** actions name can be either a verb (e.g. InitializeSimulation or UpdateParameter; this is the preferred pattern) or as a noun (named after the phenomenon it adds to the simulation, e.g. LocalEnergyMinimization or Chains).

- **Why do we split config() and run_init()?** This architecture enables easy continuation of an interrupted simulation. This feature is not currently implemented, but can be easily added upon request.

