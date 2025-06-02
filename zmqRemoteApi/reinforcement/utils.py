import numpy as np
import matplotlib.pyplot as plt
import cv2

def action_to_speed(action):
    return [
        (0.3, 0.3),  # slow forward
        (1.0, 1.0),  # fast forward
        (0.3, 0.6),  # slow left
        (0.6, 0.3),  # slow right
        (-1.0, 1.0), # spin left
        (1.0, -1.0), # spin right
        (-0.5, -0.5),
        (0.0, 0.0)
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

def plot_rewards(rewards, window=10, save_path='training_plot.png'):
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(range(window - 1, len(rewards)), smoothed, '--', label=f'{window}-Ep Moving Avg')
    plt.xlabel('Episode'); plt.ylabel('Total Reward')
    plt.grid()
    plt.legend()
    plt.title('Training Progress')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()