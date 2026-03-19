import numpy as np
import matplotlib.pyplot as plt
from grid_world import create_4x3_world, create_10x10_world, Actions
from rl_agents import QLearningAgent, SARSAAgent, ApproximateQLearning, ApproximateSARSA
from direct_utility import DirectUtilityAgent, FunctionApproximationAgent

def compare_all_methods(env, episodes=1000, max_steps=1000):
    """Compare all implemented methods on the same environment"""
    agents = {
        'Q-Learning': QLearningAgent(env),
        'SARSA': SARSAAgent(env),
        'Q-Learning (FA)': ApproximateQLearning(env),
        'SARSA (FA)': ApproximateSARSA(env),
        'Direct Utility': DirectUtilityAgent(env),
        'Value Function Approx': FunctionApproximationAgent(env)
    }
    
    returns_data = {name: [] for name in agents.keys()}
    steps_data = {name: [] for name in agents.keys()}
    
    for name, agent in agents.items():
        print(f"Training {name}...")
        
        for episode in range(episodes):
            state = env.reset()
            episode_return = 0
            steps = 0
            
            action = agent.get_action(state) if hasattr(agent, 'get_action') else None
            
            while steps < max_steps:
                if isinstance(agent, (DirectUtilityAgent, FunctionApproximationAgent)):
                    action = agent.get_action(state) if hasattr(agent, 'get_action') else None
                    next_state, reward, done = env.step(state, action)
                    agent.update(state, reward)
                else:
                    next_state, reward, done = env.step(state, action)
                    next_action = agent.get_action(next_state) if not done else None
                    agent.update(state, action, reward, next_state, next_action)
                    action = next_action
                
                episode_return += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            returns_data[name].append(episode_return)
            steps_data[name].append(steps)

            if hasattr(agent, 'epsilon'):
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    plot_comparison_results(returns_data, steps_data, env)
    plot_final_policies(agents, env)
    
    return agents

def plot_comparison_results(returns_data, steps_data, env):
    """Plot learning curves for all methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for name, returns in returns_data.items():
        ax1.plot(returns, label=name)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Learning Curves (Returns)')
    ax1.legend()
    
    for name, steps in steps_data.items():
        ax2.plot(steps, label=name)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Learning Curves (Steps)')
    ax2.legend()
    
    plt.suptitle(f'Comparison on {env.width}x{env.height} Grid World')
    plt.tight_layout()
    plt.show()

def plot_final_policies(agents, env):
    """Plot final policies for all agents"""
    n_agents = len(agents)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, (name, agent) in zip(axes, agents.items()):
        if isinstance(agent, (DirectUtilityAgent, FunctionApproximationAgent)):
            # Plot value function
            plot_value_function(agent, env, ax, name)
        else:
            # Plot policy
            plot_policy(agent, env, ax, name)
    
    plt.tight_layout()
    plt.show()

def plot_value_function(agent, env, ax, title):
    """Plot value function for utility-based agents"""
    values = np.zeros((env.width, env.height))
    for i in range(env.width):
        for j in range(env.height):
            state = (i, j)
            if isinstance(agent, DirectUtilityAgent):
                values[i, j] = agent.U.get(state, 0)
            else:
                values[i, j] = agent.predict(state)
    
    im = ax.imshow(values.T, origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

def plot_policy(agent, env, ax, title):
    """Plot policy for Q-learning and SARSA agents with color-coded arrows"""
    policy = {}
    values = {}
    
    for i in range(env.width):
        for j in range(env.height):
            state = (i, j)
            best_action = agent.get_action(state, epsilon=0)
            policy[state] = best_action
            if hasattr(agent, 'get_q_value'):
                values[state] = max([agent.get_q_value(state, a) for a in list(Actions)])
            else:
                values[state] = 0

    if values:
        min_val = min(values.values())
        max_val = max(values.values())
        norm = plt.Normalize(min_val, max_val)
        cmap = plt.cm.viridis

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.grid(True)
    
    for state, action in policy.items():
        if state not in env.obstacles and state != env.goal_pos:
            x, y = state
            dx, dy = env.moves[action]
            
            if values:
                color = cmap(norm(values[state]))
            else:
                color = 'blue'
            
            ax.arrow(x + 0.5, y + 0.5, dx*0.4, dy*0.4,
                    head_width=0.2, head_length=0.2, fc=color, ec=color)

    for obs in env.obstacles:
        ax.add_patch(plt.Rectangle(obs, 1, 1, facecolor='black'))
    ax.add_patch(plt.Rectangle(env.goal_pos, 1, 1, facecolor='green'))
    
    if values:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Q-value')
    
    ax.set_title(title)

if __name__ == "__main__":
    print("\nTesting 10x10 world with goal at (4,4) [5,5 in 1-based indexing]...")
    env_10x10 = create_10x10_world((4, 4))  # Using 0-based indexing
    agents = compare_all_methods(env_10x10) 
