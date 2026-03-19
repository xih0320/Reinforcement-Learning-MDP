import numpy as np
from wumpus import WumpusMDP
from mdp_base import FiniteStateMDP

def value_iteration(mdp, gamma=0.9, epsilon=1e-6):
    """
    Value iteration algorithm implementation
    Returns optimal value function and policy
    """
    V = {state: 0 for state in mdp.states}
    policy = {}
    
    while True:
        delta = 0
        for state in mdp.states:
            if mdp.is_terminal(state):
                continue
                
            v = V[state]
            max_value = float('-inf')
            best_action = None
            
            for action in mdp.actions_at(state):
                expected_value = 0
                for next_state, prob in mdp.p(state, action):
                    expected_value += prob * (mdp.r(state, next_state) + gamma * V[next_state])
                
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action
            
            V[state] = max_value
            policy[state] = best_action
            delta = max(delta, abs(v - V[state]))
            
        if delta < epsilon:
            break
            
    return V, policy

def modified_policy_iteration(mdp, gamma=0.9, epsilon=1e-6, k=10):
    """
    Modified policy iteration algorithm implementation
    Returns optimal value function and policy
    """
    V = {state: 0 for state in mdp.states}
    policy = {state: list(mdp.actions_at(state))[0] for state in mdp.states}
    
    while True:
        for _ in range(k):
            delta = 0
            for state in mdp.states:
                if mdp.is_terminal(state):
                    continue
                    
                v = V[state]
                action = policy[state]
                expected_value = 0
                for next_state, prob in mdp.p(state, action):
                    expected_value += prob * (mdp.r(state, next_state) + gamma * V[next_state])
                
                V[state] = expected_value
                delta = max(delta, abs(v - V[state]))
                
            if delta < epsilon:
                break
        policy_stable = True
        for state in mdp.states:
            if mdp.is_terminal(state):
                continue
                
            old_action = policy[state]
            max_value = float('-inf')
            best_action = None
            
            for action in mdp.actions_at(state):
                expected_value = 0
                for next_state, prob in mdp.p(state, action):
                    expected_value += prob * (mdp.r(state, next_state) + gamma * V[next_state])
                
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action
            
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
                
        if policy_stable:
            break
            
    return V, policy

def evaluate_algorithms(mdp, gamma=0.9, epsilon=1e-6, k=10):
    """
    Evaluate both algorithms on the given MDP
    Returns timing and convergence information
    """
    import time
    
    start_time = time.time()
    V_vi, policy_vi = value_iteration(mdp, gamma, epsilon)
    vi_time = time.time() - start_time
    
    start_time = time.time()
    V_mpi, policy_mpi = modified_policy_iteration(mdp, gamma, epsilon, k)
    mpi_time = time.time() - start_time
    
    policy_diff = sum(1 for state in mdp.states 
                     if not mdp.is_terminal(state) and policy_vi[state] != policy_mpi[state])
    
    return {
        'value_iteration_time': vi_time,
        'modified_policy_iteration_time': mpi_time,
        'policy_differences': policy_diff,
        'value_iteration_converged': True,
        'modified_policy_iteration_converged': True
    }

def create_4x3_world():
    """
    Create a 4x3 world as described in the book
    """
    mdp = WumpusMDP(4, 3, move_cost=-0.04)
    
    mdp.add_obstacle('pit', [1, 1])
    mdp.add_obstacle('pit', [1, 2])
    
    mdp.add_obstacle('goal', [3, 2])
    
    return mdp

def create_modified_wumpus_world():
    """
    Create a modified Wumpus world with:
    - Different pit probabilities
    - Different step costs
    - Gold reward G
    - Win reward W
    """
    mdp = WumpusMDP(8, 10, move_cost=-0.1)
    
    mdp.add_obstacle('wumpus', [6, 9], -100)
    mdp.add_obstacle('wumpus', [6, 8], -50)
    mdp.add_obstacle('wumpus', [6, 7], -25)
    mdp.add_obstacle('wumpus', [7, 5], -10)
    
    mdp.add_obstacle('pit', [2, 0], -1.0)
    mdp.add_obstacle('pit', [2, 1], -0.8)
    mdp.add_obstacle('pit', [2, 2], -0.5)
    mdp.add_obstacle('pit', [5, 0], -0.3)
    mdp.add_obstacle('pit', [6, 1], -0.2)
    
    mdp.add_obstacle('goal', [7, 9], 100)  # W = 100
    
    mdp.add_object('gold', [0, 9])
    mdp.add_object('gold', [7, 0])
    mdp.add_object('gold', [1, 1])
    
    mdp.add_object('immune', [6, 0])
    mdp.add_object('immune', [1, 2])
    
    return mdp

if __name__ == "__main__":
    print("Evaluating on 4x3 world:")
    mdp_4x3 = create_4x3_world()
    results_4x3 = evaluate_algorithms(mdp_4x3)
    print(results_4x3)
    print("\nEvaluating on modified Wumpus world:")
    mdp_wumpus = create_modified_wumpus_world()
    results_wumpus = evaluate_algorithms(mdp_wumpus)
    print(results_wumpus)
