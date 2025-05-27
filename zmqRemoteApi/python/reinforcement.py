import robotica
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

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
        return 0.5, 1.0
    elif action == 2:  # turn right
        return 1.0, 0.5

# --- Main ---
def main(args=None, train_mode=True):
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

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    epsilon = 0.2
    gamma = 0.9
    batch_size = 64
    update_target_every = 100

    coppelia.start_simulation()
    step_count = 0

    while coppelia.is_running():
        sonar = robot.get_sonar()
        state = np.array(sonar[:8])  # take first 8 sonar sensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # ε-greedy policy
        if random.random() < epsilon and train_mode:
            action = random.randint(0, action_dim - 1)
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
        reward = 0
        done = False
        if front_min < 0.2:
            reward = -5  # crash penalty
            done = True
        elif front_min > 0.5:
            reward = 1  # good progress
        else:
            reward = 0

        # Store in replay buffer
        buffer.push(state, action, reward, next_state, done)

        # Training step
        if train_mode and len(buffer) > batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
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
            robot.set_speed(0.0, 0.0)
            time.sleep(0.5)

    coppelia.stop_simulation()

    # Save model
    if train_mode:
        torch.save(policy_net.state_dict(), 'dqn_model.pth')

if __name__ == '__main__':
    main(train_mode=True)

