import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm

class BanditSimulator:
    def __init__(self, num_arms: int, seed: int = None):
        """Initialize bandit with A arms, each having Gaussian distribution."""
        if seed is not None:
            np.random.seed(seed)
            
        self.num_arms = num_arms
        self.means = np.random.normal(0, 1, num_arms)
        self.variance = 1.0
        
    def pull_arm(self, arm: int) -> float:
        """Pull an arm and get reward from its Gaussian distribution."""
        if not 0 <= arm < self.num_arms:
            raise ValueError(f"Invalid arm index: {arm}")
        return np.random.normal(self.means[arm], np.sqrt(self.variance))
    
    def get_parameters(self) -> np.ndarray:
        """Return the means of all arms."""
        return self.means.copy()
    
    def get_optimal_arm(self) -> int:
        """Return the index of the arm with highest mean."""
        return np.argmax(self.means)

class UCBAgent:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms
        self.reset()
        
    def reset(self):
        """Reset the agent's state."""
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        self.total_pulls = 0
        
    def select_arm(self) -> int:
        """Select arm using UCB formula."""
        if self.total_pulls < self.num_arms:
            return self.total_pulls

        exploration = np.sqrt(2 * np.log(self.total_pulls) / self.counts)
        ucb_values = self.values + exploration
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update statistics for the chosen arm."""
        self.counts[arm] += 1
        self.total_pulls += 1
        # Incremental update
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

class EpsilonGreedyAgent:
    def __init__(self, num_arms: int, epsilon: float):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.reset()
        
    def reset(self):
        """Reset the agent's state."""
        self.counts = np.zeros(self.num_arms)
        self.values = np.zeros(self.num_arms)
        
    def select_arm(self) -> int:
        """Select arm using ε-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: choose random arm
            return np.random.randint(self.num_arms)
        else:
            # Exploit: choose best arm
            return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        """Update statistics for the chosen arm."""
        self.counts[arm] += 1
        # Incremental update
        n = self.counts[arm]
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n

def run_experiment(
    agent,
    bandit: BanditSimulator,
    num_steps: int
) -> Tuple[List[float], List[bool]]:
    """Run experiment and return cumulative regret and optimal action choices."""
    optimal_arm = bandit.get_optimal_arm()
    optimal_mean = bandit.means[optimal_arm]
    
    cumulative_regret = []
    optimal_actions = []
    current_regret = 0
    
    for _ in range(num_steps):
        arm = agent.select_arm()
        reward = bandit.pull_arm(arm)
        agent.update(arm, reward)
        
        regret = optimal_mean - bandit.means[arm]
        current_regret += regret
        cumulative_regret.append(current_regret)
        
        optimal_actions.append(arm == optimal_arm)
    
    return cumulative_regret, optimal_actions

def plot_results(
    results: Dict,
    num_steps: int,
    title: str
):
    """Plot results from multiple experiments."""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    for label, (regret, _) in results.items():
        plt.plot(range(num_steps), regret, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('Cumulative Regret Over Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for label, (_, optimal) in results.items():
        window = min(100, num_steps // 10)
        optimal_array = np.array(optimal)
        cumsum = np.cumsum(np.insert(optimal_array, 0, 0))
        optimal_ma = (cumsum[window:] - cumsum[:-window]) / window
        plt.plot(range(window-1, num_steps), optimal_ma, label=label)
    
    plt.xlabel('Steps')
    plt.ylabel('Fraction of Optimal Actions (Moving Average)')
    plt.title('Optimal Action Selection Over Time')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    seeds = [42, 123, 456, 789, 101112]
    num_arms_list = [5, 10, 20]
    m_values = [2, 5, 10]
    epsilon_values = [0.1, 0.2, 0.3]
    
    for num_arms in num_arms_list:
        for m in m_values:
            num_steps = m * num_arms

            all_results = {}

            for seed in seeds:
                bandit = BanditSimulator(num_arms, seed)
                
                # UCB
                ucb_agent = UCBAgent(num_arms)
                ucb_regret, ucb_optimal = run_experiment(ucb_agent, bandit, num_steps)
                
                # ε-greedy with different ε values
                for epsilon in epsilon_values:
                    eps_agent = EpsilonGreedyAgent(num_arms, epsilon)
                    eps_regret, eps_optimal = run_experiment(eps_agent, bandit, num_steps)
                    
                    # Accumulate results
                    label = f"ε-greedy (ε={epsilon})"
                    if label not in all_results:
                        all_results[label] = (np.zeros(num_steps), np.zeros(num_steps))
                    all_results[label] = (
                        all_results[label][0] + np.array(eps_regret),
                        all_results[label][1] + np.array(eps_optimal)
                    )
                
                # Accumulate UCB results
                if "UCB" not in all_results:
                    all_results["UCB"] = (np.zeros(num_steps), np.zeros(num_steps))
                all_results["UCB"] = (
                    all_results["UCB"][0] + np.array(ucb_regret),
                    all_results["UCB"][1] + np.array(ucb_optimal)
                )
            
            # Average results over seeds
            for label in all_results:
                all_results[label] = (
                    all_results[label][0] / len(seeds),
                    all_results[label][1] / len(seeds)
                )
            
            title = f"Comparison of Algorithms (A={num_arms}, m={m})"
            plot_results(all_results, num_steps, title)

if __name__ == "__main__":
    main()
