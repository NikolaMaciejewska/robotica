import numpy as np
import matplotlib.pyplot as plt
import cv2
from mypy.build import default_data_dir

""""(0.7, 0.7),  # slow forward
(0.8, 0.8),  # medium forward
(1.0, 1.0),  # fast forward

(0.4, 0.6),  # slow left
(0.6, 0.4),  # slow right
(0.2, 0.5), # sharper left
(0.5, 0.2), # sharper right

(-0.4, -0.4),
(0.0, 0.0)"""
"""
(1.5, 1.5),  # slow forward
(2.5, 2.5),  # fast forward

(-1.5, 1.5),  # in-place left
(1.5, -1.5),  # in-place right

(1.5, 2.5),  # gentle left curve
(2.5, 1.5),  # gentle right curve"""


def action_to_speed(action):
    return [
        (1.5, 1.5),  # slow forward
        (2.5, 2.5),  # fast forward

        (-1.5, 1.5),  # in-place left
        (1.5, -1.5),  # in-place right

        (1.5, 2.5),  # gentle left curve
        (2.5, 1.5),  # gentle right curve

        (-0.5, -0.5)
    ][action]

def detect_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ball_detected = False
    ball_x_norm = 0.0
    ball_radius_norm = 0.0
    if contours:
        ((x, _), radius) = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        if radius > 5:
            center_x = img.shape[1] / 2
            ball_x_norm = (x - center_x) / center_x
            ball_radius_norm = radius / center_x
            ball_detected = True
    return ball_detected, ball_x_norm, ball_radius_norm

def plot_rewards(rewards, window=10, save_path='training_plot1.png'):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')

    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window - 1, len(rewards)), smoothed, '--', label=f'{window}-Ep Moving Avg')
    else:
        print(f"Not enough rewards to compute a moving average with window={window}. Skipping moving average plot.")

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.legend()
    plt.title('Training Progress')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()