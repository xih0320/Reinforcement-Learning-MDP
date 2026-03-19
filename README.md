# Reinforcement Learning & MDP Algorithms

## Highlights
- Implemented 8 reinforcement learning and MDP algorithms from scratch  
- Compared tabular and approximate methods across multiple environments  
- Analyzed convergence behavior and exploration strategies in stochastic settings  

---

## Overview
This project implements and benchmarks classical reinforcement learning and Markov Decision Process (MDP) algorithms in Python.

The goal is to understand how different learning paradigms — exact methods, model-free learning, and function approximation — perform under uncertainty and stochastic transitions.

Algorithms implemented:
- Value Iteration  
- Modified Policy Iteration  
- Q-Learning  
- SARSA  
- Approximate Q-Learning  
- Approximate SARSA  
- Direct Utility Estimation  
- UCB vs ε-Greedy (multi-armed bandits)  

---

## Why it matters
Sequential decision-making under uncertainty is a core problem in many real-world systems.

This project demonstrates how different reinforcement learning strategies handle:
- Exploration vs exploitation trade-offs  
- Convergence speed vs stability  
- Generalization with function approximation  

Applications include:
- Recommendation systems  
- Online advertising optimization  
- Robotics and control systems  

---

## Environments

### Grid World
Stochastic grid navigation with slip probability.

- 4×3 world (classic benchmark)
- 10×10 world with varied goal positions  

---

### Wumpus World
Custom MDP environment with:
- Pits, Wumpus agents, gold, immunity objects  
- Stochastic transitions (80/10/10 probabilities)  

---

## Results

### Key Findings
- Value Iteration and Policy Iteration converge fastest but require full model knowledge  
- Q-Learning and SARSA successfully learn optimal policies from experience  
- Function approximation improves scalability but introduces instability  
- UCB outperforms ε-Greedy in early exploration efficiency  

---

### Learning Curves
![4x3 Comparison](Comparison_on_4_3_Grid_World.png)

![10x10 Comparison](Comparison_on_10_10.png)

---

### Policy Visualization
![6 Methods](Comparison_with_6_methods.png)

---

### Reward Curves
![Reward 5x5](Reward_5_5.png)

---

## File Structure

```
├── mdp_base.py
├── wumpus_mdp.py
├── wumpus_demo.py
├── grid_world.py
├── value_iteration.py
├── rl_agents.py
├── direct_utility.py
├── comparison.py
├── bandit_simulator.py
└── bandit_ucb_vs_epsilon.py
```
---

## Tech Stack
Python · NumPy · Matplotlib · tqdm · typing  

---

## Key Design Patterns
- Unified `FiniteStateMDP` interface across environments  
- Shared base classes for tabular and approximate agents  
- Modular training and evaluation pipeline (`train_agent`, `compare_agents`)  

---

## Key Takeaways
- Exact methods are optimal but not scalable  
- Model-free methods learn from interaction without environment knowledge  
- Function approximation enables scaling but requires careful tuning  
- Exploration strategy significantly impacts early learning performance  
