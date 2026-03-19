import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Actions(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class GridWorld:
    def __init__(self, width, height, obstacles=None, goal_pos=None, goal_reward=1.0, 
                 move_cost=-0.04, slip_prob=0.1):
        self.width = width
        self.height = height
        self.obstacles = obstacles or []
        self.goal_pos = goal_pos or (width-1, height-1)
        self.goal_reward = goal_reward
        self.move_cost = move_cost
        self.slip_prob = slip_prob
        self.moves = {
            Actions.UP: (0, 1),
            Actions.RIGHT: (1, 0),
            Actions.DOWN: (0, -1),
            Actions.LEFT: (-1, 0)
        }
        
    def is_valid_pos(self, pos):
        x, y = pos
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                pos not in self.obstacles)
    
    def get_next_pos(self, pos, action):
        if action is None:
            return pos  # 如果没有动作，保持在当前位置
        dx, dy = self.moves[action]
        next_pos = (pos[0] + dx, pos[1] + dy)
        return next_pos if self.is_valid_pos(next_pos) else pos
    
    def step(self, pos, action):
        if pos == self.goal_pos:
            return pos, self.goal_reward, True
            
        if action is None:
            return pos, self.move_cost, False

        next_pos = self.get_next_pos(pos, action)

        slip_left = (action.value - 1) % 4
        slip_right = (action.value + 1) % 4
        
        probs = [1 - 2*self.slip_prob, self.slip_prob, self.slip_prob]
        actions = [action, Actions(slip_left), Actions(slip_right)]
        
        chosen_action = np.random.choice(actions, p=probs)
        final_pos = self.get_next_pos(pos, chosen_action)
        
        reward = self.goal_reward if final_pos == self.goal_pos else self.move_cost
        
        return final_pos, reward, final_pos == self.goal_pos
    
    def reset(self):
        return (0, 0)
    
    def get_state_features(self, pos):
        """Get features for function approximation"""
        x, y = pos
        features = np.array([
            x / (self.width - 1),
            y / (self.height - 1),
            abs(x - self.goal_pos[0]) / (self.width - 1),
            abs(y - self.goal_pos[1]) / (self.height - 1),
            *[1 if pos == obs else 0 for obs in self.obstacles]
        ])
        return features
    
    def visualize_policy(self, policy, title="Policy"):
        """Visualize policy using arrows"""
        fig, ax = plt.subplots(figsize=(8, 8))

        for i in range(self.width):
            for j in range(self.height):
                pos = (i, j)
                if pos in self.obstacles:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor='black'))
                elif pos == self.goal_pos:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor='green'))
                else:
                    action = policy.get(pos, None)
                    if action is not None:
                        dx, dy = self.moves[action]
                        ax.arrow(i + 0.5, j + 0.5, dx*0.4, dy*0.4, 
                                head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(range(self.width + 1))
        ax.set_yticks(range(self.height + 1))
        ax.grid(True)
        ax.set_title(title)
        plt.show()

def create_4x3_world():
    """Create 4x3 world as described in Chapter 17"""
    world = GridWorld(4, 3, move_cost=-0.04)
    world.obstacles = [(1, 1)]
    world.goal_pos = (3, 2)    
    world.goal_reward = 1.0
    return world

def create_10x10_world(goal_pos=(9, 9)):
    """Create 10x10 world with goal at specified position"""
    world = GridWorld(10, 10, move_cost=-0.04)
    world.goal_pos = goal_pos
    world.goal_reward = 1.0
    return world 
