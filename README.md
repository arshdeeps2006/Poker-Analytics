# Poker AI & Decision Modeling

An algorithmic game theory project exploring **Counterfactual Regret Minimization (CFR)** and Monte Carlo methods to develop poker agents with near-Nash-equilibrium strategies.

## Overview

This project focuses on building and evaluating poker decision-making engines using regret minimization, strategic optimization, and exploitability analysis. Starting with simplified poker variants, the goal is to develop a foundation for more complex imperfect-information games.

## Key Features

- **Counterfactual Regret Minimization (CFR):** Implemented CFR to approximate Nash equilibria in imperfect-information poker games.
- **Monte Carlo CFR:** Engineered External Sampling Monte Carlo CFR to reduce the state-space traversal requirements by **80%** in Kuhn and Leduc poker.
- **Exploitability Analysis:** Evaluated learned strategies using empirical Best Response computations to measure vulnerability to strategic deviations.
- **Gameplay Evaluation:** Simulated extensive gameplay against baseline heuristic agents to assess strategic performance.
- **Convergence Analysis:** Evaluated convergence toward Nash-equilibrium strategies using exploitability and expected-value metrics.

## Results

| Metric | Result |
|---|---|
| Kuhn Poker Nash Bound | -0.056 EV |
| Leduc Poker Nash Bound | -0.086 EV |
| State-Space Reduction | 80% |
| Win-Rate Performance | +1.25 EV against heuristic opponents |
| Execution Time | Sub-second runtimes |

The learned strategies demonstrated negligible exploitability and consistently outperformed baseline heuristic opponents in the evaluated environments.

## Algorithms & Concepts

- Counterfactual Regret Minimization (CFR)
- External Sampling Monte Carlo CFR
- Nash Equilibrium Approximation
- Best Response Computation
- Exploitability Evaluation
- Imperfect-Information Games
- Expected Value (EV) Analysis

## Poker Variants

**Kuhn Poker**  
A simplified poker environment used to study regret minimization, strategy convergence, and equilibrium computation.

**Leduc Poker**  
A more complex poker variant used to evaluate the algorithms under a larger game tree and a richer strategic environment.

## Future Directions

- Extend the framework toward larger poker variants.
- Investigate abstraction techniques for reducing computational complexity.
- Explore stronger evaluation methods and diverse opponent strategies.
- Work toward a Texas Hold'em AI using scalable CFR-based algorithms.

---

*Focused on algorithmic game theory, regret minimization, and strategic decision-making under uncertainty.*
