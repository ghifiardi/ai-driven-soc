"""
Simple Cybersecurity Defense Simulation - NO TRAINING REQUIRED
Just demonstrates the environment and a rule-based agent
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ========== Custom Gym Environment ==========
class CyberDefenseEnv(gym.Env):
    """Custom Environment simulating network defense against cyber attacks"""
    
    def __init__(self, num_nodes=5):
        super(CyberDefenseEnv, self).__init__()
        self.num_nodes = num_nodes
        self.max_steps = 30
        self.current_step = 0
        self.observation_space = spaces.Box(low=0, high=10, shape=(num_nodes + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.node_statuses = np.zeros(self.num_nodes)
        self.attack_type = 0
        self.threat_level = 0
        return self._get_state()
    
    def _get_state(self):
        return np.concatenate([self.node_statuses, [self.attack_type, self.threat_level]]).astype(np.float32)
    
    def _simulate_attack(self):
        if np.random.random() < 0.4:
            self.attack_type = np.random.randint(1, 4)
            self.threat_level = np.random.randint(3, 11)
            target_node = np.random.randint(0, self.num_nodes)
            if self.node_statuses[target_node] == 0:
                self.node_statuses[target_node] = 1
        
        for i in range(self.num_nodes):
            if self.node_statuses[i] == 1 and np.random.random() < 0.3:
                self.node_statuses[i] = 2
    
    def step(self, action):
        self.current_step += 1
        reward = 0
        
        if action == 0:  # Monitor
            if self.threat_level > 5:
                reward += 2
        elif action == 1:  # Block traffic
            if self.attack_type == 2:
                reward += 10
                self.threat_level = max(0, self.threat_level - 5)
                self.node_statuses[self.node_statuses == 1] = 0
            elif self.threat_level < 3:
                reward -= 1
        elif action == 2:  # Patch system
            if self.attack_type in [1, 3]:
                reward += 10
                self.threat_level = max(0, self.threat_level - 4)
                self.node_statuses[self.node_statuses == 1] = 0
            elif self.threat_level < 3:
                reward -= 1
        elif action == 3:  # Isolate node
            if np.any(self.node_statuses == 2):
                reward += 8
                self.node_statuses[self.node_statuses == 2] = 0
            else:
                reward -= 2
        elif action == 4:  # No action
            if self.threat_level > 7:
                reward -= 5
        
        current_compromised = np.sum(self.node_statuses == 2)
        reward -= current_compromised * 20
        
        self._simulate_attack()
        
        if np.all(self.node_statuses == 0):
            reward += 5
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {
            'compromised_nodes': int(current_compromised),
            'threat_level': self.threat_level
        }


# ========== Simple Rule-Based Agent ==========
class RuleBasedAgent:
    """Simple rule-based agent that uses heuristics instead of ML"""
    
    def __init__(self):
        self.action_names = ['Monitor', 'Block Traffic', 'Patch System', 'Isolate Node', 'No Action']
    
    def act(self, state):
        """Make decisions based on rules"""
        num_nodes = len(state) - 2
        node_statuses = state[:num_nodes]
        attack_type = int(state[-2])
        threat_level = int(state[-1])
        
        # Priority 1: Isolate compromised nodes
        if np.any(node_statuses == 2):
            return 3  # Isolate
        
        # Priority 2: Handle specific attack types
        if attack_type == 2 and threat_level > 4:  # DDoS
            return 1  # Block traffic
        
        if attack_type in [1, 3] and threat_level > 4:  # Malware or Intrusion
            return 2  # Patch system
        
        # Priority 3: Monitor if threat is present
        if threat_level > 5:
            return 0  # Monitor
        
        # Default: No action
        return 4


# ========== Simulation Functions ==========
def run_simulation(agent, episodes=5, verbose=True):
    """Run simulation episodes"""
    env = CyberDefenseEnv(num_nodes=5)
    all_rewards = []
    all_actions = []
    
    action_names = ['Monitor', 'Block Traffic', 'Patch System', 'Isolate Node', 'No Action']
    attack_names = ['None', 'Malware', 'DDoS', 'Intrusion']
    
    print("=" * 80)
    print(f"Running {episodes} simulation episodes")
    print("=" * 80 + "\n")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_actions = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"EPISODE {episode + 1}")
            print(f"{'='*80}")
        
        for step in range(env.max_steps):
            action = agent.act(state)
            episode_actions.append(action)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            if verbose and step < 10:  # Show first 10 steps
                print(f"\nStep {step + 1:2d}:")
                print(f"  Attack: {attack_names[int(state[-2])]:12s} | Threat Level: {int(state[-1]):2d}")
                print(f"  Action: {action_names[action]:15s} | Reward: {reward:6.1f}")
                print(f"  Compromised Nodes: {info['compromised_nodes']}")
            
            state = next_state
            
            if done:
                break
        
        all_rewards.append(total_reward)
        all_actions.extend(episode_actions)
        
        print(f"\n{'='*80}")
        print(f"Episode {episode + 1} Summary: Total Reward = {total_reward:.2f}")
        
        # Count actions
        action_counts = [episode_actions.count(i) for i in range(5)]
        print(f"Actions taken:")
        for i, (name, count) in enumerate(zip(action_names, action_counts)):
            print(f"  {name:15s}: {count:3d} times")
        print(f"{'='*80}")
    
    return all_rewards, all_actions


def plot_results(rewards, actions):
    """Plot simulation results"""
    action_names = ['Monitor', 'Block\nTraffic', 'Patch\nSystem', 'Isolate\nNode', 'No\nAction']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Rewards per episode
    axes[0].bar(range(1, len(rewards) + 1), rewards, color='steelblue', alpha=0.7)
    axes[0].axhline(y=np.mean(rewards), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(rewards):.2f}')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('Rewards per Episode', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Action distribution
    action_counts = [actions.count(i) for i in range(5)]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#95a5a6']
    axes[1].bar(range(5), action_counts, color=colors, alpha=0.7)
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(action_names, fontsize=10)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Action Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Action percentage pie chart
    axes[2].pie(action_counts, labels=action_names, autopct='%1.1f%%', 
                colors=colors, startangle=90)
    axes[2].set_title('Action Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cyber_defense_simulation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results saved to 'cyber_defense_simulation.png'")
    plt.show()


# ========== Main Execution ==========
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "CYBERSECURITY DEFENSE SIMULATION")
    print(" " * 25 + "(Rule-Based Agent Demo)")
    print("=" * 80)
    
    print("\nAction Space:")
    print("  0 = Monitor       - Watch for threats")
    print("  1 = Block Traffic - Block DDoS attacks")
    print("  2 = Patch System  - Fix malware/intrusions")
    print("  3 = Isolate Node  - Isolate compromised nodes")
    print("  4 = No Action     - Do nothing\n")
    
    print("State Space:")
    print("  - Node Status: 0=Secure, 1=Vulnerable, 2=Compromised")
    print("  - Attack Type: 0=None, 1=Malware, 2=DDoS, 3=Intrusion")
    print("  - Threat Level: 0-10\n")
    
    # Create rule-based agent
    agent = RuleBasedAgent()
    
    # Run simulation
    rewards, actions = run_simulation(agent, episodes=5, verbose=True)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("OVERALL SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Total Episodes: {len(rewards)}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Best Episode: {max(rewards):.2f}")
    print(f"Worst Episode: {min(rewards):.2f}")
    print(f"Total Actions Taken: {len(actions)}")
    print("=" * 80)
    
    # Plot results
    plot_results(rewards, actions)
    
    print("\n✓ Simulation complete!")
    print("  Check 'cyber_defense_simulation.png' for visualization")



















