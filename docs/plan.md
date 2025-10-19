# JAX Spiel Conversion Plan

## Vision
Re-implement core OpenSpiel concepts using JAX-friendly data structures and functional patterns while maintaining API familiarity.

## Milestones
1. **Core API Skeleton**
   - Define JAX-compatible `Game` and `State` abstractions with immutable dataclasses.
   - Provide utilities for batched operations and JIT compilation.
2. **Example Game Implementations**
   - Port simple matrix game (e.g., Matching Pennies) as functional JAX code.
   - Extend to sequential games (e.g., Kuhn Poker) using pure functions.
3. **Policy & Value Functions**
   - Implement policy evaluation using JAX transformations (jit, vmap).
   - Provide example RL algorithms (policy gradients).
4. **Benchmarking & Testing**
   - Mirror selected OpenSpiel tests using pytest + JAX.
   - Add performance benchmarks.

## Current Iteration Goals
- Establish package structure (`jax_spiel`).
- Define `TensorGame` and `TensorState` to represent matrix games.
- Implement Matching Pennies as an initial conversion target.
- Provide JAX-compatible payoff computation and best response utility.
- Cover functionality with unit tests using TDD.

## Future Considerations
- Support extensive-form games with tree-based state representation.
- Integrate with Optax/Flax for learning algorithms.
- Add documentation and tutorials.
