"""
Cybersecurity Defense Agent using OpenAI Gym and Deep Q-Network (DQN)
QUICK DEMO VERSION - Fewer episodes for faster execution

This implements a reinforcement learning agent that learns to defend a network
from cyber attacks by taking defensive actions like blocking, monitoring, and patching.
"""

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ========== Custom Gym Environment ==========
class CyberDefenseEnv(gym.Env):
    """
    Custom Environment simulating network defense against cyber attacks.
    
    State: [node_0_status, node_1_status, ..., attack_type, threat_level]
    - node_status: 0=secure, 1=vulnerable, 2=compromised
    - attack_type: 0=none, 1=malware, 2=ddos, 3=intrusion
    - threat_level: 0-10 (current threat intensity)
    
    Actions: 
    0=monitor, 1=block_traffic, 2=patch_system, 3=isolate_node, 4=no_action
    
    Rewards:
    +10 for preventing compromise, -20 for compromised node, 
    -1 for false positives, -5 for missed threats
    """
    
    def __init__(self, num_nodes=5):
        super(CyberDefenseEnv, self).__init__()
        
        self.num_nodes = num_nodes
        self.max_steps = 50  # Reduced from 100 for quick demo
        self.current_step = 0
        
        # State: node statuses + attack type + threat level
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(num_nodes + 2,), dtype=np.float32
        )
        
        # Actions: monitor, block, patch, isolate, no_action
        self.action_space = spaces.Discrete(5)
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.node_statuses = np.zeros(self.num_nodes)  # All secure initially
        self.attack_type = 0  # No attack
        self.threat_level = 0
        self.compromised_count = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Return current state observation"""
        return np.concatenate([
            self.node_statuses,
            [self.attack_type, self.threat_level]
        ]).astype(np.float32)
    
    def _simulate_attack(self):
        """Simulate random cyber attacks"""
        # Random chance of new attack
        if np.random.random() < 0.3:
            self.attack_type = np.random.randint(1, 4)  # 1=malware, 2=ddos, 3=intrusion
            self.threat_level = np.random.randint(3, 11)
            
            # Make random nodes vulnerable
            target_node = np.random.randint(0, self.num_nodes)
            if self.node_statuses[target_node] == 0:
                self.node_statuses[target_node] = 1  # Make vulnerable
        
        # Attacks may escalate vulnerable nodes to compromised
        for i in range(self.num_nodes):
            if self.node_statuses[i] == 1 and np.random.random() < 0.4:
                self.node_statuses[i] = 2  # Compromise node
                self.compromised_count += 1
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.current_step += 1
        reward = 0
        
        # Apply agent's defensive action
        if action == 0:  # Monitor
            if self.threat_level > 5:
                reward += 2  # Good monitoring during high threat
        
        elif action == 1:  # Block traffic
            if self.attack_type == 2:  # DDoS attack
                reward += 10
                self.threat_level = max(0, self.threat_level - 5)
                # Restore vulnerable nodes
                self.node_statuses[self.node_statuses == 1] = 0
            elif self.threat_level < 3:
                reward -= 1  # False positive penalty
        
        elif action == 2:  # Patch system
            if self.attack_type in [1, 3]:  # Malware or intrusion
                reward += 10
                self.threat_level = max(0, self.threat_level - 4)
                # Restore vulnerable nodes
                self.node_statuses[self.node_statuses == 1] = 0
            elif self.threat_level < 3:
                reward -= 1
        
        elif action == 3:  # Isolate node
            if np.any(self.node_statuses == 2):  # Has compromised nodes
                reward += 8
                # Restore compromised nodes (isolated and cleaned)
                self.node_statuses[self.node_statuses == 2] = 0
            else:
                reward -= 2  # Unnecessary isolation
        
        elif action == 4:  # No action
            if self.threat_level > 7:
                reward -= 5  # Missed critical threat
        
        # Penalty for compromised nodes
        current_compromised = np.sum(self.node_statuses == 2)
        reward -= current_compromised * 20
        
        # Simulate new attacks
        self._simulate_attack()
        
        # Bonus for keeping all nodes secure
        if np.all(self.node_statuses == 0):
            reward += 5
        
        done = self.current_step >= self.max_steps
        
        return self._get_state(), reward, done, {
            'compromised_nodes': int(current_compromised),
            'threat_level': self.threat_level
        }


# ========== DQN Neural Network ==========
class DQN(nn.Module):
    """Deep Q-Network for learning optimal defense policy"""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)  # Reduced network size
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ========== DQN Agent ==========
class DQNAgent:
    """DQN Agent with experience replay and target network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Reduced memory size
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay
        self.learning_rate = 0.001
        self.batch_size = 32  # Smaller batch size
        
        # Main network and target network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.update_target_model()
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def replay(self):
        """Train on batch of experiences from memory"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first for efficiency
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ========== Training Function ==========
def train_agent(episodes=100, render=False):
    """Train the DQN agent"""
    env = CyberDefenseEnv(num_nodes=5)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    avg_scores = []
    
    action_names = ['Monitor', 'Block Traffic', 'Patch System', 'Isolate Node', 'No Action']
    
    print(f"Using device: {agent.device}")
    print(f"State size: {state_size}, Action size: {action_size}\n")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        actions_taken = [0] * action_size
        
        for time_step in range(env.max_steps):
            action = agent.act(state)
            actions_taken[action] += 1
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network every 5 episodes
        if episode % 5 == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-20:])  # Average of last 20 episodes
        avg_scores.append(avg_score)
        
        if episode % 10 == 0:
            print(f"Episode: {episode:3d}/{episodes} | Score: {total_reward:7.2f} | "
                  f"Avg Score: {avg_score:7.2f} | Epsilon: {agent.epsilon:.3f}")
            print(f"  Actions: {dict(zip(action_names, actions_taken))}")
    
    return agent, scores, avg_scores


# ========== Evaluation Function ==========
def evaluate_agent(agent, episodes=5):
    """Evaluate trained agent"""
    env = CyberDefenseEnv(num_nodes=5)
    total_rewards = []
    action_names = ['Monitor', 'Block Traffic', 'Patch System', 'Isolate Node', 'No Action']
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        actions_taken = [0] * 5
        
        for _ in range(env.max_steps):
            action = agent.act(state)
            actions_taken[action] += 1
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"\nEval Episode {episode + 1}: Reward = {total_reward:.2f}")
        print(f"  Actions Distribution: {dict(zip(action_names, actions_taken))}")
    
    print(f"\n{'='*60}")
    print(f"Average Evaluation Reward: {np.mean(total_rewards):.2f}")
    print(f"Std Deviation: {np.std(total_rewards):.2f}")
    print(f"{'='*60}")
    return total_rewards


# ========== Main Execution ==========
if __name__ == "__main__":
    print("=" * 60)
    print("Cybersecurity Defense Agent - DQN Training (Quick Demo)")
    print("=" * 60)
    print("Training with 100 episodes (quick demo version)")
    print("=" * 60 + "\n")
    
    # Train agent
    agent, scores, avg_scores = train_agent(episodes=100)
    
    print("\n" + "=" * 60)
    print("Training Complete! Evaluating Agent...")
    print("=" * 60)
    
    # Evaluate
    eval_rewards = evaluate_agent(agent, episodes=5)
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores, alpha=0.4, color='blue', label='Episode Reward')
    plt.plot(avg_scores, label='Moving Average (20 episodes)', linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(eval_rewards)), eval_rewards, color='green', alpha=0.7)
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Reward')
    plt.title('Evaluation Performance')
    plt.axhline(y=np.mean(eval_rewards), color='r', linestyle='--', 
                label=f'Mean: {np.mean(eval_rewards):.2f}')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Show learning curve
    window = 10
    smoothed = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
    plt.plot(smoothed, linewidth=2, color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.title(f'Learning Curve (Window={window})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cyber_defense_results_quick.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results plot saved as 'cyber_defense_results_quick.png'")
    print("\nSimulation Complete!")
    plt.show()



















