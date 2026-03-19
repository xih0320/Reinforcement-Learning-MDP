import numpy as np
from grid_world import GridWorld, Actions
import matplotlib.pyplot as plt

class DirectUtilityAgent:
    """Direct Utility Estimation Agent - learns state utilities directly from experience"""
    def __init__(self, env: GridWorld, learning_rate=0.1, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.U = {}
        self.visit_counts = {}
        self.returns = {}
        
    def get_action(self, state, epsilon=None):
        """Random action selection (exploration only)"""
        return np.random.choice(list(Actions))
    
    def update(self, state, reward):
        """Update utility estimate for a state"""
        if state not in self.visit_counts:
            self.visit_counts[state] = 0
            self.returns[state] = 0
            self.U[state] = 0
            
        self.visit_counts[state] += 1
        self.returns[state] += reward
        self.U[state] = self.returns[state] / self.visit_counts[state]

class FunctionApproximationAgent:
    """Function approximation for value function estimation"""
    def __init__(self, env: GridWorld, learning_rate=0.01, gamma=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_features = 12
        self.weights = np.zeros(self.num_features)
        
    def get_features(self, state):
        """Get enhanced feature vector for a state"""
        if state is None:
            return np.zeros(self.num_features)
            
        x, y = state
        goal_x, goal_y = self.env.goal_pos
        
        manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
        euclidean_dist = ((x - goal_x)**2 + (y - goal_y)**2)**0.5
        
        features = [
            1.0,
            x / self.env.width,
            y / self.env.height,
            manhattan_dist / (self.env.width + self.env.height),
            euclidean_dist / ((self.env.width**2 + self.env.height**2)**0.5),
            float(x == goal_x),
            float(y == goal_y),
            float((x, y) in self.env.obstacles),
            np.exp(-manhattan_dist / 5),
            np.sin(np.pi * x / self.env.width),
            np.cos(np.pi * x / self.env.width),
            np.sin(np.pi * y / self.env.height)
        ]
        return np.array(features)
    
    def predict(self, state):
        """Predict value for a state"""
        if state is None:
            return 0.0
        features = self.get_features(state)
        return np.dot(self.weights, features)
    
    def get_action(self, state):
        """Get best action based on current value function"""
        if state is None:
            return None
            
        best_value = float('-inf')
        best_action = None
        
        for action in Actions:
            next_state, reward, done = self.env.step(state, action)
            value = reward
            if not done:
                value += self.gamma * self.predict(next_state)
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
    
    def update(self, state, reward, next_state=None, done=False):
        """Update weights using TD learning"""
        if state is None:
            return
            
        features = self.get_features(state)
        prediction = self.predict(state)
        
        if done or next_state is None:
            target = reward
        else:
            next_value = self.predict(next_state)
            target = reward + self.gamma * next_value
            
        error = target - prediction
        self.weights += self.learning_rate * error * features

def train_and_compare(env, episodes=1000, max_steps=1000):
    """Train and compare Direct Utility and Function Approximation agents"""
    # Initialize agents
    direct_agent = DirectUtilityAgent(env)
    fa_agent = FunctionApproximationAgent(env)
    
    direct_returns = []
    fa_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_return = 0
        
        for step in range(max_steps):
            action = direct_agent.get_action(state)
            next_state, reward, done = env.step(state, action)
            episode_return += reward
            direct_agent.update(state, reward)
            state = next_state
            if done:
                break
        direct_returns.append(episode_return)
        
        state = env.reset()
        episode_return = 0
        
        for step in range(max_steps):
            action = fa_agent.get_action(state)
            next_state, reward, done = env.step(state, action)
            episode_return += reward
            fa_agent.update(state, reward, next_state, done)
            state = next_state
            if done:
                break
        fa_returns.append(episode_return)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(direct_returns, label='Direct Utility')
    plt.plot(fa_returns, label='Function Approximation')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curves')
    plt.legend()
    

    plt.subplot(122)
    plot_value_functions(env, direct_agent, fa_agent)
    plt.show()
    
    return direct_agent, fa_agent

def plot_value_functions(env, direct_agent, fa_agent):
    """Plot value function comparisons"""
    x = np.linspace(0, env.width-1, env.width)
    y = np.linspace(0, env.height-1, env.height)
    X, Y = np.meshgrid(x, y)
    Z_direct = np.zeros_like(X)
    Z_fa = np.zeros_like(X)
    
    for i in range(env.width):
        for j in range(env.height):
            state = (i, j)
            Z_direct[j, i] = direct_agent.U.get(state, 0)
            Z_fa[j, i] = fa_agent.predict(state)
    
    plt.imshow(Z_direct - Z_fa)
    plt.colorbar(label='Value Difference')
    plt.title('Value Function Difference\n(Direct - Function Approximation)')

if __name__ == "__main__":
    # Test on 4x3 world
    env_4x3 = create_4x3_world()
    direct_agent_4x3, fa_agent_4x3 = train_and_compare(env_4x3)
    
    # Test on 10x10 worlds
    env_10x10_far = create_10x10_world((9, 9))
    direct_agent_10x10_far, fa_agent_10x10_far = train_and_compare(env_10x10_far)
    
    env_10x10_mid = create_10x10_world((4, 4))
    direct_agent_10x10_mid, fa_agent_10x10_mid = train_and_compare(env_10x10_mid) 
