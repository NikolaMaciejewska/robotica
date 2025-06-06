import robotica, time, torch, numpy as np
from config import *
from model import DQN
from memory import ReplayBuffer
from utils import action_to_speed, detect_ball, plot_rewards
import torch.optim as optim
import torch.nn.functional as F
import random

def get_state(robot, obstacle_mode, ball_mode, last_ball_seen=False, last_ball_dir=0.0, last_ball_radius=0.0):
    sonar = robot.get_sonar()
    image_error = False
    try:
        img = robot.get_image()
        ball_detected, ball_x_norm, ball_radius_norm = detect_ball(img)
    except Exception as e:
        print("Image read error:", e)
        image_error = True

    state = []
    if obstacle_mode: state.extend(sonar[:8])
    if ball_mode: state += [ball_x_norm, ball_radius_norm]

    # Add memory info
    state.append(float(last_ball_seen))
    state.append(last_ball_dir)
    state.append(last_ball_radius)

    # (
    return np.array(state, dtype=np.float32), ball_detected, ball_x_norm, ball_radius_norm, sonar, image_error

def compute_reward(obstacle_mode, ball_mode, sonar_next, ball_detected, ball_x_norm, ball_radius_norm, action, last_ball_seen=False, last_ball_dir=0.0, last_ball_radius=0.0):
    reward = 0.0
    sonar_min = min(sonar_next)
    too_close = sonar_min < 0.1

    # Penalize crashes
    if obstacle_mode and too_close:
        return -10.0

    # Ball-related behavior (only if safe)
    if ball_mode and not too_close:
        if not ball_detected:
            reward -= 4.0

            if last_ball_seen:
                if abs(last_ball_dir) > 0.2:
                    going_left = action in [2, 4]
                    going_right = action in [3, 5]

                    if last_ball_dir < 0:
                        if going_left:
                            reward += 1.0
                        elif going_right:
                            reward -= 0.5
                    elif last_ball_dir > 0:
                        if going_right:
                            reward += 1.0
                        elif going_left:
                            reward -= 0.5
                else:
                    if action in [0, 1]:
                        reward += 1.0
            else:
                # Ball wurde noch nie gesehen
                if action in [2, 3]:
                    reward += 0.3
        else:
            reward += 1.0
            reward += (1.0 - abs(ball_x_norm)) * 2.0
            reward += ball_radius_norm * 5.0

            # Bonus for centered and forward
            if abs(ball_x_norm) < 0.2 and action == 1:
                reward += 1.0

            # Extra reward if getting closer
            if ball_radius_norm > last_ball_radius + 0.01:
                reward += 1.0

    return reward

def main(args=None, train_mode=False, continue_training = False, obstacle_mode=True, ball_mode=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = (STATE_DIM_OBSTACLE if obstacle_mode else 0) + (STATE_DIM_BALL if ball_mode else 0) + 3
    epsilon = EPSILON_START
    last_ball_seen = False
    last_ball_dir = 0.0  # Range: -1 (left) to 1 (right)
    last_ball_radius = 0.0

    policy_net, target_net = DQN(state_dim, ACTION_DIM).to(device), DQN(state_dim, ACTION_DIM).to(device)
    if not train_mode and not continue_training:
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    elif continue_training:
        print("Continuing training from existing model...")
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        epsilon = 0.7
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
        state, ball_detected, ball_x_norm, ball_radius_norm, sonar, image_error = get_state(robot, obstacle_mode, ball_mode, last_ball_seen, last_ball_dir, last_ball_radius)
        if image_error:
            print("image error")
            continue

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        print("ball_detected: ", ball_detected)
        # If ball is seen, store direction
        if ball_detected:
            last_ball_seen = True
            last_ball_dir = ball_x_norm
            last_ball_radius = ball_radius_norm
        else:
            last_ball_seen = False

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
        v_left = robot.sim.getJointVelocity(robot.left_motor)
        v_right = robot.sim.getJointVelocity(robot.right_motor)
        print("Left motor velocity:", v_left)
        print("Right motor velocity:", v_right)
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

        orient = robot.get_orientation()
        upside_down = (abs(orient[0]) > 1.0 or abs(orient[1]) > 1.0)
        print("UPSIDE DOWN", upside_down)

        if done or upside_down:
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
    main(train_mode=False, continue_training=False, obstacle_mode=True, ball_mode=True)