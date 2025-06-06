import time
import random
import torch
import torch.nn.functional as F
import numpy as np

from config import *
from utils import action_to_speed, detect_ball, plot_rewards
from model import DQN
from memory import ReplayBuffer
from main import get_state, compute_reward  # Hilfsfunktionen aus main.py importieren
import robotica

def check_upside_down(robot):
    orient = robot.get_orientation()
    return abs(orient[0]) > 1.0 or abs(orient[1]) > 1.0

def train_phase_1(obstacle_mode, ball_mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = (STATE_DIM_OBSTACLE if obstacle_mode else 0) + (STATE_DIM_BALL if ball_mode else 0) + 3
    action_dim = ACTION_DIM

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer()

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()

    epsilon = EPSILON_START
    step_count = 0
    episode_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        robot.reset_to_initial_position()
        last_ball_seen = False
        last_ball_dir = 0.0
        last_ball_radius = 0.0
        current_reward = 0.0
        steps_since_seen = 0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            state, ball_detected, ball_x_norm, ball_radius_norm, sonar, image_error = get_state(
                robot, obstacle_mode, ball_mode, last_ball_seen, last_ball_dir, last_ball_radius
            )
            if image_error:
                continue

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            if ball_detected:
                last_ball_seen = True
                last_ball_dir = ball_x_norm
                last_ball_radius = ball_radius_norm
                steps_since_seen = 0
            else:
                last_ball_seen = False
                steps_since_seen += 1

            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            print("ACTION: ", action)

            lspeed, rspeed = action_to_speed(action)
            robot.set_speed(lspeed, rspeed)
            time.sleep(0.1)

            next_state, next_ball_detected, next_ball_x, next_ball_radius, sonar_next, image_error = get_state(
                robot, obstacle_mode, ball_mode, last_ball_seen, last_ball_dir, last_ball_radius
            )
            if image_error:
                continue

            reward = compute_reward(
                obstacle_mode, ball_mode, sonar_next,
                next_ball_detected, next_ball_x, next_ball_radius,
                action, last_ball_seen, last_ball_dir, last_ball_radius
            )
            print("REWARD: ", reward)
            current_reward += reward

            done = (step == MAX_STEPS_PER_EPISODE)
            buffer.push(state, action, reward, next_state, done)

            if len(buffer) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
                expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                loss = F.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if step_count % UPDATE_TARGET_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

            step_count += 1

            #front_min = min(sonar_next[2:6])
            if steps_since_seen > MAX_STEPS_WITHOUT_BALL or check_upside_down(robot):
                print(f"Respawning robot at episode {episode}, step {step} (ball lost, collision or upside down)")
                robot.reset_to_initial_position()
                last_ball_seen = False
                last_ball_dir = 0.0
                last_ball_radius = 0.0
                steps_since_seen = 0

            if done:
                episode_rewards.append(current_reward)
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
                print(f"Episode {episode} finished with reward {current_reward:.2f}, epsilon {epsilon:.3f}")

                # Durchschnitt der letzten REWARD_WINDOW_SIZE Episoden
                if len(episode_rewards) >= REWARD_WINDOW_SIZE:
                    avg_reward = sum(episode_rewards[-REWARD_WINDOW_SIZE:]) / REWARD_WINDOW_SIZE
                    print(f"Average reward over last {REWARD_WINDOW_SIZE} episodes: {avg_reward:.2f}")
                    if avg_reward >= TARGET_REWARD:
                        print("Training goal reached! Aborting training early.")
                        break
                break

    coppelia.stop_simulation()
    torch.save(policy_net.state_dict(), MODEL_PATH)
    np.save(REWARDS_PATH, episode_rewards)
    plot_rewards(episode_rewards, save_path=PLOT_PATH)

if __name__ == '__main__':
    train_phase_1(obstacle_mode=True, ball_mode=True)