# wiggin

## Introductoin
wiggin is an experimental high-level modular framework for building polymer simulations.

wiggin's key idea is that any simulation can be decomposed into as a series of atomistic "actions". Such actions can declare or modify some aspect of the system or perform any general task or computation. E.g., an action can initialize a simulation object, calculate positions of loops, add a force that forms chains or loops, compute simulation steps, or save data. The goal of this framework is to enable combinatorial combinations of different blocks of code without making the final code too heavy.

Each action must be  must be configured once, before running. After configuration, actions can be performed either once, when the simulation is initialized (at the "init" stage"), or, at every block of the simulation (the "loop" stage).



## Technical notes

### Execution order:
- add actions
- configure actions
- execute all run_init() once 
- execute all run_loop(), repeatedly

### Structure of a SimConstructor

- actions
- config
- config['shared']. Reserved keys:
    - name - name of the simulation. Can be generated from the non-default parameters using auto_name()
    - N - the number of particles in the simulation
    - initial_conformation - the initial coordinates of all particles in the system
    - folder - the folder to store the configuration and the output of the simulation
- config['actions'].

## Q&A
- **What is the preferred naming pattern for actions?** actions name can be either a verb (e.g. InitializeSimulation or UpdateParameter; this is the preferred pattern) or as a noun (named after the phenomenon it adds to the simulation, e.g. LocalEnergyMinimization or Chains).

- **Why do we split config() and run_init()?**