# PSOAS
Particle Swarm Optimization Assisted by Surrogates

# 1. Introduction

Our goal is to create a software library, which solves global optimization problems. The solver will be a modified and hybridized version of the "Particle Swarm Optimization"-algorithm (PSO). PSO is an heuristic approach in which a set of particle is scattered in the search space. Each particle is determined by its position, velocity and personal best position. Furthermore the particles communicate with each other and exchange their personal optima. From one step to the next, the velocity of a particle is updated using their current velocity and/or position additionally to their personal best position and the best position that the particle found through communication with other particles.

Interesting points to address would be the velocity update equation and the usage of some kind of surrogate model to use past function evaluations and guide the swarm to exploration or exploitation. The latter could especially be helpful to prevent function evaluations where other particles have already been. Additionally, local models could be applied in densely sampled regions. For simple models, calculating the optimum should be easier and this might boost the convergence of the algorithm.  
