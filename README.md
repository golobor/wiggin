# wiggin
wiggin is an experimental high-level modular framework for building polymer simulations.

wiggin's key idea is that any simulation can be decomposed into as a series of atomistic "actions". Such actions can declare or modify some aspect of the system or perform any general task or computation. E.g., an action can initialize a simulation object, calculate positions of loops, add a force that forms chains or loops, compute simulation steps, or save data. The goal of this framework is to enable combinatorial combinations of different blocks of code without making the final code too heavy.

Each action must be  must be configured once, before running. After configuration, actions can be performed either once, when the simulation is initialized (at the "init" stage"), or, at every block of the simulation (the "loop" stage).
