from grid_world import GridWorld, create_4x3_world, create_10x10_world
import numpy as np
from grid_world import Actions, GridWorld
from typing import Dict, Tuple, Callable
import matplotlib.pyplot as plt

class TabularAgent:
    def __init__(self, env: GridWorld, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        
    def _ensure_state_exists(self, state):
        """Ensure state exists in Q-table"""
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in Actions}
            
    def get_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
            
        self._ensure_state_exists(state)
            
        if np.random.random() < epsilon:
            return np.random.choice(list(Actions))
        return max(self.Q[state].items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state, next_action=None):
        """Update Q-values (implemented by subclasses)"""
        raise NotImplementedError

class QLearningAgent(TabularAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)
            
        max_next_q = max(self.Q[next_state].values())
        current_q = self.Q[state][action]
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

class SARSAAgent(TabularAgent):
    def update(self, state, action, reward, next_state, next_action):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        next_q = self.Q[next_state][next_action] if next_action is not None else 0
        current_q = self.Q[state][action]
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )

class ApproximateAgent:
    def __init__(self, env: GridWorld, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.basic_feature_dim = 8
        self.action_dim = 4
        feature_dim = self.basic_feature_dim + self.action_dim
        self.weights = np.zeros(feature_dim)
        
    def get_state_features(self, state):
        """Get the enhanced state features"""
        x, y = state
        goal_x, goal_y = self.env.goal_pos
        
        manhattan_dist = abs(x - goal_x) + abs(y - goal_y)
        euclidean_dist = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        
        features = [
            x / self.env.width,
            y / self.env.height,
            manhattan_dist / (self.env.width + self.env.height),
            euclidean_dist / np.sqrt(self.env.width**2 + self.env.height**2),  #
            float(x == goal_x),
            float(y == goal_y),
            np.exp(-manhattan_dist / 5),
            float((x, y) in self.env.obstacles)
        ]
        return np.array(features)
        
    def get_q_value(self, state, action):
        """The Q-value is calculated using the enhanced features"""
        if action is None:
            return 0.0
            
        state_features = self.get_state_features(state)
        action_features = np.zeros(self.action_dim)
        action_features[action.value] = 1
        combined_features = np.concatenate([state_features, action_features])
        
        return np.dot(self.weights, combined_features)
    
    def get_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            return np.random.choice(list(Actions))
            
        q_values = [self.get_q_value(state, a) for a in Actions]
        return Actions(np.argmax(q_values))
    
    def update_weights(self, state, action, target):
        """Update weights using gradient descent"""
        if action is None:
            return
        state_features = self.get_state_features(state)
        action_features = np.zeros(self.action_dim)
        action_features[action.value] = 1
        combined_features = np.concatenate([state_features, action_features])
        prediction = self.get_q_value(state, action)
        error = target - prediction
        self.weights += self.alpha * error * combined_features

class ApproximateQLearning(ApproximateAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        if next_action is None:
            target = reward
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in Actions]
            max_next_q = max(next_q_values)
            target = reward + self.gamma * max_next_q
            
        self.update_weights(state, action, target)

class ApproximateSARSA(ApproximateAgent):
    def update(self, state, action, reward, next_state, next_action):
        if next_action is None:
            target = reward
        else:
            next_q = self.get_q_value(next_state, next_action)
            target = reward + self.gamma * next_q
            
        self.update_weights(state, action, target)

def train_agent(agent, env: GridWorld, episodes=1000, max_steps=1000):
    """Train an agent and return learning curves"""
    steps_per_episode = []
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        action = agent.get_action(state)
        
        while not done and steps < max_steps:
            next_state, reward, done = env.step(state, action)
            total_reward += reward
            next_action = agent.get_action(next_state) if not done else None

            agent.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            steps += 1
            
        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    return steps_per_episode, rewards_per_episode

def plot_learning_curves(steps_data, rewards_data, labels, title):
    """Plot learning curves for multiple agents"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for steps, label in zip(steps_data, labels):
        ax1.plot(steps, label=label)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps per Episode')
    ax1.legend()
    
    for rewards, label in zip(rewards_data, labels):
        ax2.plot(rewards, label=label)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward per Episode')
    ax2.legend()
    
    plt.suptitle(title)
    plt.show()

def compare_agents(env: GridWorld, episodes=1000):
    """Compare performance of all agents on given environment"""
    agents = [
        QLearningAgent(env),
        SARSAAgent(env),
        ApproximateQLearning(env),
        ApproximateSARSA(env)
    ]
    labels = ['Q-Learning (Tabular)', 'SARSA (Tabular)', 
              'Q-Learning (Approx)', 'SARSA (Approx)']
    
    steps_data = []
    rewards_data = []
    
    for agent in agents:
        steps, rewards = train_agent(agent, env, episodes)
        steps_data.append(steps)
        rewards_data.append(rewards)
        policy = {state: agent.get_action(state, epsilon=0) 
                 for state in [(i, j) for i in range(env.width) 
                             for j in range(env.height)]}
        env.visualize_policy(policy, title=f"{labels[agents.index(agent)]} Policy")
    
    plot_learning_curves(steps_data, rewards_data, labels, 
                        f"Learning Curves - {env.width}x{env.height} Grid World")

if __name__ == "__main__":
    env_4x3 = create_4x3_world()
    compare_agents(env_4x3)
    env_10x10_1 = create_10x10_world((9, 9))
    compare_agents(env_10x10_1)

    env_10x10_2 = create_10x10_world((4, 4))
    compare_agents(env_10x10_2) 
