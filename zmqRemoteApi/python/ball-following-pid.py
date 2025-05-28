import robotica
import time
import cv2
import numpy as np

# PID parameters for wall following
Kp = 0.9
Ki = 0.0
Kd = 0.3

target_distance = 0.3
max_speed = 1.5
min_speed = -0.5
base_speed = 1.3

# PID parameters for ball tracking
ball_Kp = 0.007
ball_Ki = 0.0005
ball_Kd = 0.003

initial_search_time = 10.0
ball_lost_timeout = 3.0


def pid_wall_follow(error, last_error, integral, dt):
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

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

    if min_front < 0.3 or left < 0.1 or right < 0.1:
        steer_strength = 2.0
    elif min_front < 0.45:
        steer_strength = 1.0
    else:
        return None

    left_speed = base_speed + steer_strength
    right_speed = base_speed - steer_strength
    return left_speed, right_speed

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

    while coppelia.is_running():
        current_time = time.time()
        dt = current_time - last_time
        dt = max(dt, 0.05)
        last_time = current_time

        readings = robot.get_sonar()
        img = robot.get_image()
        ball = find_red_ball(img)

        obstacle = avoid(readings)

        if ball:
            ball_last_seen = time.time()
            x, y, r = ball
            img_center = img.shape[1] // 2
            error = img_center - x

            correction, ball_integral = pid_ball_tracking(error, ball_last_error, ball_integral, dt)
            ball_last_error = error

            speed = base_speed * 0.5 if r > 40 else base_speed
            left_speed = speed - correction
            right_speed = speed + correction

            if obstacle:
                left_speed, right_speed = obstacle

            print(f"Tracking ball | Speeds: {left_speed:.2f}, {right_speed:.2f}")

        elif obstacle:
            left_speed, right_speed = obstacle
            print("Avoiding obstacle")

        elif not initial_search_done:
            if time.time() - initial_search_start < initial_search_time:
                left_speed, right_speed =1.0, -1.0
                print("Initial ball search")
            else:
                initial_search_done = True
                print("Switching to wall following")
                continue

        elif ball_last_seen is None or time.time() - ball_last_seen > ball_lost_timeout:
            side_distance = readings[1]
            error = target_distance - side_distance
            correction, wall_integral = pid_wall_follow(error, wall_last_error, wall_integral, dt)
            wall_last_error = error
            left_speed = base_speed + correction
            right_speed = base_speed - correction
            print("Wall following")

        else:
            left_speed, right_speed = 0.5, -0.5  # spinning to relocate ball
            print("Searching for ball after loss")

        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))

        robot.set_speed(left_speed, right_speed)

        if ball:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        time.sleep(0.05)

    coppelia.stop_simulation()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()