from abc import ABC, abstractmethod

class MDPState(ABC):
    """Abstract base class for MDP states"""
    @abstractmethod
    def clone(self):
        """Return a deep copy of this state"""
        pass

    @abstractmethod
    def __hash__(self):
        """Return a hash of this state"""
        pass

    @abstractmethod
    def __eq__(self, other):
        """Compare this state with another"""
        pass

class FiniteStateMDP(ABC):
    """Abstract base class for finite state MDPs"""
    
    @property
    @abstractmethod
    def states(self):
        """Return a list of all possible states"""
        pass

    @property
    @abstractmethod
    def actions(self):
        """Return a list of all possible actions"""
        pass

    @property
    @abstractmethod
    def initial_state(self):
        """Return the initial state"""
        pass

    @abstractmethod
    def actions_at(self, state):
        """Return list of possible actions at given state"""
        pass

    @abstractmethod
    def p(self, state, action):
        """Return list of (next_state, probability) pairs for given state and action"""
        pass

    @abstractmethod
    def r(self, state, next_state):
        """Return reward for transition from state to next_state"""
        pass

    @abstractmethod
    def is_terminal(self, state):
        """Return whether state is terminal"""
        pass

def modified_policy_iteration(mdp, gamma=0.9, eval_iters=5):
    """
    Modified policy iteration algorithm for MDPs.

    Args:
        mdp: The MDP to solve
        gamma: Discount factor
        eval_iters: Number of partial policy evaluation iterations per policy improvement

    Returns:
        U: Dictionary mapping states to their utilities
        policy: Dictionary mapping states to optimal actions
    """
    U = {}
    policy = {}
    for state in mdp.states:
        U[state] = 0
        actions = mdp.actions_at(state)
        if actions:
            policy[state] = actions[0]

    stable = False
    while not stable:
        for _ in range(eval_iters):
            U_new = U.copy()
            for state in U.keys():
                if mdp.is_terminal(state):
                    U_new[state] = 0  # Terminal states have a fixed utility
                    continue
                a = policy.get(state)
                if a is not None:
                    expected_utility = sum(
                        p * (mdp.r(state, s_next) + gamma * U[s_next])
                        for s_next, p in mdp.p(state, a)
                    )
                    U_new[state] = expected_utility
            U = U_new

        stable = True
        for state in U.keys():
            if mdp.is_terminal(state):
                continue
            best_action = None
            best_value = float("-inf")
            for a in mdp.actions_at(state):
                expected_utility = sum(
                    p * (mdp.r(state, s_next) + gamma * U[s_next])
                    for s_next, p in mdp.p(state, a)
                )
                if expected_utility > best_value:
                    best_value = expected_utility
                    best_action = a
            if policy.get(state) != best_action:
                policy[state] = best_action
                stable = False

    return U, policy
