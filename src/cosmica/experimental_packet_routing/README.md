# Routing formulation

## Static network

### Definitions

- Definition of sets
    - $`\mathcal{N}`$: Set of network nodes
    - $`\mathcal{K}`$: Set of communication requests
- Definition of parameters
    - $`C_{ij} \geq 0`$: Communication capacity from node $`i`$ to node $`j`$ in bps.
    - $`d_{ik} \in \mathbb{R}`$: Information demand of request $`k`$
        - If there's a communication request $`k`$ of amount $`\delta`$ from node $`i`$ to node $`j`$, then $`d_{ik} = -\delta`$ and $`d_{jk} = \delta`$.
- Design variables
    - $`x_{ijk} \geq 0`$: Communication flow of request $`k`$ from node $`i`$ to node $`j`$.

![Static network](../../img/static_network.drawio.svg)

### Optimization problem formulation

It can be formualted as a linear programming problem:

Minimize

- $`\sum_{i,j \in \mathcal{N} \colon i \neq j} \sum_{k \in \mathcal{K}} x_{ijk}`$

Subject to

- $`\forall i,j \in \mathcal{N} (i \neq j) \colon \sum_{k \in \mathcal{K}} x_{ijk} \leq C_{ij}`$
- $`\forall i \in \mathcal{N}, k \in \mathcal{K} \colon \sum_{k \in \mathcal{K}} \left( x_{jik} - x_{ijk} \right) \geq d_{ik}`$

### Ideas for improvement

- We can penalize the communication requests that are not satisfied, instead of just strictly applying the constraint.

## Dynamic network

Ideas:

- Stack the static network along the time axis.
- Add a network edge from node $`i`$ at time $`t`$ to node $`i`$ at time $`t+1`$, representing a data storage in node $`i`$.
