import robotica, time, torch, numpy as np
from config import *
from model import DQN
from memory import ReplayBuffer
from utils import action_to_speed, detect_ball, plot_rewards
import torch.optim as optim
import torch.nn.functional as F
import random


def get_state(robot, obstacle_mode, ball_mode):
    sonar = robot.get_sonar()
    img = robot.get_image()
    ball_detected, ball_x_norm, ball_radius_norm = detect_ball(img)
    state = []
    if obstacle_mode: state.extend(sonar[:8])
    if ball_mode: state += [ball_x_norm, ball_radius_norm]
    return np.array(state, dtype=np.float32), ball_detected, sonar

def compute_reward(obstacle_mode, ball_mode, sonar_next, ball_detected, ball_x_norm, ball_radius_norm, action):
    reward = 0.0
    front_min = min(sonar_next[2:6])
    if obstacle_mode:
        if front_min < 0.1:
            reward -= 6.0
        elif front_min > 0.25:
            reward += 1.0
    if ball_mode:
        if not ball_detected:
            reward -= 5.0
        else:
            reward += (1.0 - abs(ball_x_norm)) * 2.0
            reward += ball_radius_norm * 1.0
    reward += 0.3 if (action == 0 or action == 1) else -0.15
    return reward

def main(args=None, train_mode=False, continue_training = False, obstacle_mode=True, ball_mode=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = (STATE_DIM_OBSTACLE if obstacle_mode else 0) + (STATE_DIM_BALL if ball_mode else 0)
    epsilon = EPSILON_START

    policy_net, target_net = DQN(state_dim, ACTION_DIM).to(device), DQN(state_dim, ACTION_DIM).to(device)
    if not train_mode and not continue_training:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    elif continue_training:
        print("Continuing training from existing model...")
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        epsilon = 0.2
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer()

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    start_time = time.time()
    step_count = 0
    episode_rewards, current_reward = [], 0

    while coppelia.is_running() and (time.time() - start_time < MAX_DURATION):
        state, ball_detected, sonar = get_state(robot, obstacle_mode, ball_mode)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        print("ball_detected: ", ball_detected)

        if random.random() < epsilon and train_mode:
            action = random.randint(0, ACTION_DIM - 1)
            # action with highest Q-value predicted
        else:
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
        print("ACTION: ", action)
        lspeed, rspeed = action_to_speed(action)
        robot.set_speed(lspeed, rspeed)
        time.sleep(0.1)

        next_state, next_ball_detected, sonar_next = get_state(robot, obstacle_mode, ball_mode)
        reward = compute_reward(obstacle_mode, ball_mode, sonar_next, next_ball_detected, next_state[-2], next_state[-1], action)
        print("REWARD: ", reward)
        done = step_count % MAX_STEPS_PER_EPISODE == 0
        current_reward += reward

        buffer.push(state, action, reward, next_state, done)

        if train_mode and len(buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states, next_states = torch.FloatTensor(states).to(device), torch.FloatTensor(next_states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards, dones = torch.FloatTensor(rewards).unsqueeze(1).to(device), torch.FloatTensor(dones).unsqueeze(1).to(device)
            q_values = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].detach().unsqueeze(1)
            expected_q = rewards + GAMMA * next_q * (1 - dones)
            loss = F.mse_loss(q_values, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step_count % UPDATE_TARGET_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_rewards.append(current_reward)
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            print("Episode: {}, Reward: {:.2f}, Epsilon: {:.2f}".format(len(episode_rewards), current_reward, epsilon))
            current_reward = 0
            robot.reset_to_initial_position()

        step_count += 1

    coppelia.stop_simulation()

    if train_mode:
        torch.save(policy_net.state_dict(), MODEL_PATH)
        np.save(REWARDS_PATH, episode_rewards)
        plot_rewards(episode_rewards, save_path=PLOT_PATH)

if __name__ == '__main__':
    main(train_mode=True, continue_training=False, obstacle_mode=True, ball_mode=True)