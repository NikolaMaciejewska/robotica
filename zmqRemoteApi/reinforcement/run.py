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

def run(obstacle_mode=True, ball_mode=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dim = (STATE_DIM_OBSTACLE if obstacle_mode else 0) + (STATE_DIM_BALL if ball_mode else 0) + 3

    # Load trained model
    policy_net = DQN(state_dim, ACTION_DIM).to(device)
    policy_net.load_state_dict(torch.load('models_archive/dqn_model_phase1_try1.pth', map_location=device))
    policy_net.eval()

    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', True)
    coppelia.start_simulation()
    start_time = time.time()
    step_count = 0

    last_ball_seen = False
    last_ball_dir = 0.0
    last_ball_radius = 0.0

    while coppelia.is_running() and (time.time() - start_time < MAX_DURATION):
        state, ball_detected, ball_x_norm, ball_radius_norm, sonar, image_error = get_state(
            robot, obstacle_mode, ball_mode, last_ball_seen, last_ball_dir, last_ball_radius
        )
        if image_error:
            print("Image error. Skipping step.")
            continue

        if ball_detected:
            last_ball_seen = True
            last_ball_dir = ball_x_norm
            last_ball_radius = ball_radius_norm
        else:
            last_ball_seen = False

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        lspeed, rspeed = action_to_speed(action)
        robot.set_speed(lspeed, rspeed)

        orient = robot.get_orientation()
        if abs(orient[0]) > 1.0 or abs(orient[1]) > 1.0:
            print("Robot flipped. Resetting position.")
            robot.reset_to_initial_position()

        time.sleep(0.1)
        step_count += 1

    coppelia.stop_simulation()

if __name__ == '__main__':
    run(obstacle_mode=True, ball_mode=True)