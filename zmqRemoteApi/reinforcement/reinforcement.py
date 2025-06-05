import robotica
import numpy as np
import time
import cv2
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
    if action == 0:  # forward
        return 1.0, 1.0
    elif action == 1:  # slight left
        return 0.5, 1.0
    elif action == 2:  # slight right
        return 1.0, 0.5
    elif action == 3:  # hard left
        return -1.0, 1.0
    elif action == 4:  # hard right
        return 1.0, -1.0

def detect_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ball_detected = False
    ball_x_norm = 0.0
    ball_radius_norm = 0.0

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            img_center_x = img.shape[1] / 2
            ball_x_norm = (x - img_center_x) / img_center_x  # -1 to +1
            ball_radius_norm = radius / img_center_x         # relative size
            ball_detected = True

    return ball_detected, ball_x_norm, ball_radius_norm

import matplotlib.pyplot as plt

def plot_rewards(rewards, window=10, save_path='training_plot1.png'):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(range(window - 1, len(rewards)), smoothed, label=f'{window}-Episode Moving Average', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --- Main ---
def main(args=None, train_mode=False, obstacle_mode=True, ball_mode=True):
    max_duration = 600  # z. B. 10 Minuten Training
    max_steps_per_episode = 100
    start_time = time.time()

    episode_rewards = []
    current_reward = 0

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)

    # Setup DQN
    state_dim = 0
    if obstacle_mode:
        state_dim += 8  # Assume 8 sonar sensors (Pioneer P3DX has 16 but usually you can pick 8 important ones)
    if ball_mode:
        state_dim += 2

    action_dim = 5  # forward, left, right

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    if not train_mode:
        policy_net.load_state_dict(torch.load('models_archive/dqn_model1.pth', map_location=device))
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
        img = robot.get_image()
        ball_detected, ball_x_norm, ball_radius_norm = detect_ball(img)
        state = []
        if obstacle_mode:
            state.extend(sonar[:8])  # add sonar readings
        if ball_mode:
            state.append(ball_x_norm)
            state.append(ball_radius_norm)
        state = np.array(state, dtype=np.float32)
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
        print("ACTION: ", action)

        # Execute action
        lspeed, rspeed = action_to_speed(action)
        robot.set_speed(lspeed, rspeed)
        time.sleep(0.1)

        # New observation
        sonar_next = robot.get_sonar()
        img = robot.get_image()
        ball_detected, ball_x_norm, ball_radius_norm = detect_ball(img)
        next_state = []
        if obstacle_mode:
            next_state.extend(sonar_next[:8])  # add sonar readings
        if ball_mode:
            next_state.append(ball_x_norm)
            next_state.append(ball_radius_norm)

        next_state = np.array(next_state, dtype=np.float32)

        # Reward design
        front_min = min(sonar_next[2], sonar_next[3], sonar_next[4], sonar_next[5])
        done = False

        reward = 0.0

        # --- Obstacle avoidance (high priority) ---
        if obstacle_mode:
            if front_min < 0.1:
                reward -= 5.0
                # optional Recovery Move:
                robot.set_speed(-0.5, -0.5)
                time.sleep(0.3)
                robot.set_speed(0.5, -0.5)
                time.sleep(0.3)
                #done = True  # End episode
            elif front_min > 0.2:
                reward += 1.0  # Reward for staying safe

        # --- Ball tracking (high priority) ---
        if ball_mode:
            if not ball_detected:
                reward -= 5.0  # Penalty for losing sight of the ball
            else:
                # Encourage centering
                reward += (1.0 - abs(ball_x_norm)) * 2.0
                # Encourage approaching
                reward += ball_radius_norm * 1.0

        # --- General movement incentives (low priority) ---
        if action == 0:
            reward += 0.3  # small bonus for moving forward
        else:
            reward -= 0.15  # small penalty for turning
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
        if step_count % max_steps_per_episode == 0:
            done = True

        if step_count % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_rewards.append(current_reward)
            current_reward = 0
            robot.reset_to_initial_position()

    coppelia.stop_simulation()

    # Save model
    if train_mode:
        torch.save(policy_net.state_dict(), 'models_archive/dqn_model1.pth')
        np.save('models_archive/episode_rewards_phase1_try1.npy', episode_rewards)
        plot_rewards(episode_rewards)

if __name__ == '__main__':
    main(train_mode=True, obstacle_mode=True, ball_mode=True)

