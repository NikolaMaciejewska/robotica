import robotica
import time
import cv2
import numpy as np

# PID parameters
Kp = 0.9
Ki = 0.0
Kd = 0.3

ball_Kp = 0.007
ball_Ki = 0.0005
ball_Kd = 0.003

target_distance = 0.1
max_speed = 3.5
min_speed = -0.5
base_speed = 2.0

initial_search_time = 10.0
ball_lost_timeout = 3.0

def pid_wall_follow(error, last_error, integral, dt):
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

def perform_wall_following(readings, last_error, integral, last_time):
    target_distance = 0.3
    base_speed = 1.3
    max_speed = 1.5
    min_speed = -0.5

    def avoid_local(readings):
        front_left = readings[3]
        front_right = readings[4]
        min_front = min(front_left, front_right)

        if min_front < 0.25:
            steer_strength = 2.0
        elif min_front < 0.4:
            steer_strength = 1.0
        else:
            return None

        left_speed = base_speed + steer_strength
        right_speed = base_speed - steer_strength
        return left_speed, right_speed

    obstacle = avoid_local(readings)
    if obstacle:
        return obstacle[0], obstacle[1], last_error, integral, time.time()

    side_distance = readings[1]
    error = target_distance - side_distance

    current_time = time.time()
    dt = current_time - last_time
    dt = max(dt, 0.1)
    last_time = current_time

    correction, integral = pid_wall_follow(error, last_error, integral, dt)
    last_error = error

    left_speed = base_speed + correction
    right_speed = base_speed - correction

    # Clamp speeds
    left_speed = max(min_speed, min(max_speed, left_speed))
    right_speed = max(min_speed, min(max_speed, right_speed))

    return left_speed, right_speed, last_error, integral, last_time


def pid_ball_tracking(error, last_error, integral, dt):
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = ball_Kp * error + ball_Ki * integral + ball_Kd * derivative
    return output, integral

def avoid(readings):
    front_left = readings[3]
    front_right = readings[4]
    right = readings[6]
    left = readings[1]

    min_front = min(front_left, front_right)
    too_close_front = min_front < 0.35
    too_close_right = right < 0.25
    too_close_left = left < 0.25

    if too_close_front and too_close_right and too_close_left:
        return 0.0, 0.0

    if too_close_front:
        if left > right:
            return base_speed - 1.5, base_speed + 1.5
        else:
            return base_speed + 1.5, base_speed - 1.5

    if too_close_right:
        return base_speed - 1.0, base_speed + 1.0

    if too_close_left:
        return base_speed + 1.0, base_speed - 1.0

    return None


def find_red_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 120, 70])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 | mask2
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                param1=50, param2=20, minRadius=5, maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return tuple(map(int, circles[0][0]))
    return None

def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    coppelia.start_simulation()

    wall_integral = 0.0
    wall_last_error = 0.0
    ball_integral = 0.0
    ball_last_error = 0.0

    last_time = time.time()
    ball_last_seen = None
    initial_search_start = time.time()
    initial_search_done = False

    last_ball_direction = 0
    just_avoided_obstacle = False
    obstacle_cleared_time = 0

    while coppelia.is_running():
        current_time = time.time()
        dt = current_time - last_time
        dt = max(dt, 0.05)
        last_time = current_time

        readings = robot.get_sonar()
        img = robot.get_image()
        ball = find_red_ball(img)

        # 1. Obstacle avoidance always first
        obstacle = avoid(readings)
        if obstacle:
            left_speed, right_speed = obstacle
            robot.set_speed(left_speed, right_speed)
            just_avoided_obstacle = True
            obstacle_cleared_time = time.time()
            print("Avoiding obstacle")
            time.sleep(0.05)
            continue

        # 2. Ball tracking
        if ball:
            ball_last_seen = time.time()
            just_avoided_obstacle = False
            x, y, r = ball
            img_center = img.shape[1] // 2
            error = img_center - x

            if error < -20:
                last_ball_direction = -1
            elif error > 20:
                last_ball_direction = 1
            else:
                last_ball_direction = 0

            correction, ball_integral = pid_ball_tracking(error, ball_last_error, ball_integral, dt)
            ball_last_error = error

            speed = base_speed * 0.6 if r > 50 else base_speed * 0.8
            left_speed = speed - correction
            right_speed = speed + correction

            print(f"Tracking ball | Speeds: {left_speed:.2f}, {right_speed:.2f}")

        # 3. Recovery after obstacle
        elif just_avoided_obstacle and (time.time() - obstacle_cleared_time < 2.0):
            if last_ball_direction == -1:
                left_speed, right_speed = -0.6, 0.6
                print("Looking left for ball after avoidance")
            elif last_ball_direction == 1:
                left_speed, right_speed = 0.6, -0.6
                print("Looking right for ball after avoidance")
            else:
                left_speed, right_speed = 0.5, -0.5
                print("Looking around for ball after avoidance")

        # 4. Initial search for ball
        elif not initial_search_done:
            if time.time() - initial_search_start < initial_search_time:
                left_speed, right_speed = -1.0, 1.0
                print("Initial ball search")
            else:
                initial_search_done = True
                print("Switching to wall following")
                continue

        # 5. Wall following if ball lost
        elif ball_last_seen is None or time.time() - ball_last_seen > ball_lost_timeout:
            left_speed, right_speed, wall_last_error, wall_integral, last_time = perform_wall_following(
                readings, wall_last_error, wall_integral, last_time
            )
            print("Wall following")


        # 6. Ball lost but recently seen – rotate to search
        else:
            left_speed, right_speed = 0.5, -0.5
            print("Searching for ball after loss")

        # Clamp speeds
        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))
        robot.set_speed(left_speed, right_speed)

        # Display ball if found
        if ball:
            x, y, r = ball
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        time.sleep(0.05)

    coppelia.stop_simulation()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
