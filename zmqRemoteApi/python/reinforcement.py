import robotica
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- DQN network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay memory for experience replay ---
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --- Action mapping ---
def action_to_speed(action):
    if action == 0:  # move forward
        return 1.0, 1.0
    elif action == 1:  # turn left
        return -1.0, 1.0
    elif action == 2:  # turn right
        return 1.0, -1.0

# --- Main ---
def main(args=None, train_mode=True):
    max_duration = 600  # z. B. 10 Minuten Training
    start_time = time.time()

    episode_rewards = []
    current_reward = 0

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')

    # Setup DQN
    state_dim = 8  # Assume 8 sonar sensors (Pioneer P3DX has 16 but usually you can pick 8 important ones)
    action_dim = 3  # forward, left, right

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    if not train_mode:
        policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=device))
        target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    epsilon = 0.2
    gamma = 0.9
    batch_size = 64
    update_target_every = 100

    coppelia.start_simulation()
    step_count = 0

    while coppelia.is_running() and (time.time() - start_time < max_duration):
        sonar = robot.get_sonar()
        state = np.array(sonar[:8])  # take first 8 sonar sensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # ε-greedy policy
            # exploration
        if random.random() < epsilon and train_mode:
            action = random.randint(0, action_dim - 1)
            # action with highest Q-value predicted
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # Execute action
        lspeed, rspeed = action_to_speed(action)
        robot.set_speed(lspeed, rspeed)
        time.sleep(0.1)

        # New observation
        sonar_next = robot.get_sonar()
        next_state = np.array(sonar_next[:8])

        # Reward design
        front_min = min(sonar_next[3], sonar_next[4])
        done = False
        if front_min < 0.2:
            reward = -5  # crash penalty
            done = True
        elif front_min > 0.5:
            reward = 1  # good progress
        else:
            reward = 0

        if action == 0:  # moving forward
            reward += 0.3  # stronger bonus for going straight
        else:
            reward -= 0.15  # stronger penalty for turning

        print(reward)
        current_reward += reward

        # Store in replay buffer
        buffer.push(state, action, reward, next_state, done)

        # Training step (if enough samples are collected)
        if train_mode and len(buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
            # bellman update
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target network update
        step_count += 1
        if step_count % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_rewards.append(current_reward)
            current_reward = 0
            robot.set_speed(0.0, 0.0)
            time.sleep(0.5)

    coppelia.stop_simulation()

    # Save model
    if train_mode:
        torch.save(policy_net.state_dict(), 'dqn_model.pth')

        # Optional smoothing
        def moving_average(data, window_size=10):
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Plot total reward per episode
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label="Episode Reward")
        plt.plot(moving_average(episode_rewards), label="Smoothed", linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()

if __name__ == '__main__':
    main(train_mode=False)

