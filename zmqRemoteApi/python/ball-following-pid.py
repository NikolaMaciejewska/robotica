import robotica
import time
import cv2
import numpy as np

# PID parameters for ball tracking
ball_Kp = 0.005
ball_Ki = 0.0003
ball_Kd = 0.0015

base_speed = 1.5
max_speed = 3.0
min_speed = -0.5

def pid_ball_tracking(error, last_error, integral, dt):
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = ball_Kp * error + ball_Ki * integral + ball_Kd * derivative
    return output, integral

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

def avoid_obstacles(readings):
    front_left = readings[3]
    front_right = readings[4]
    left = readings[1]
    right = readings[6]

    # Early detection thresholds
    front_threshold = 0.5
    side_threshold = 0.35

    # Check for obstacles ahead
    if front_left < front_threshold or front_right < front_threshold:
        if left > right:
            # Turn left
            return base_speed - 1.0, base_speed + 1.0
        else:
            # Turn right
            return base_speed + 1.0, base_speed - 1.0

    # Check for obstacles on the sides
    if left < side_threshold:
        # Turn right
        return base_speed + 0.5, base_speed - 0.5
    if right < side_threshold:
        # Turn left
        return base_speed - 0.5, base_speed + 0.5

    return None

def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    coppelia.start_simulation()

    ball_integral = 0.0
    ball_last_error = 0.0
    last_time = time.time()
    search_step = 0
    search_start_time = 0

    while coppelia.is_running():
        current_time = time.time()
        dt = max(current_time - last_time, 0.05)
        last_time = current_time

        readings = robot.get_sonar()
        img = robot.get_image()
        ball = find_red_ball(img)

        obstacle = avoid_obstacles(readings)
        if obstacle:
            left_speed, right_speed = obstacle
            robot.set_speed(left_speed, right_speed)
            print("Avoiding obstacle")
            time.sleep(0.05)
            continue

        if ball:
            search_step = 0
            search_start_time = 0

            x, y, r = ball
            img_center = img.shape[1] // 2
            error = img_center - x

            correction, ball_integral = pid_ball_tracking(error, ball_last_error, ball_integral, dt)
            ball_last_error = error

            # Dynamic speed based on distance (approximated by radius r)
            if r < 15:
                speed = 3.0
            elif r < 25:
                speed = 2.4
            elif r < 35:
                speed = 1.8
            elif r < 45:
                speed = 1.0
            else:
                speed = 0.6

            if abs(correction) > 0.6:
                speed *= 0.6
            elif abs(correction) > 0.3:
                speed *= 0.8

            left_speed = speed - correction
            right_speed = speed + correction

            print(f"Tracking ball (r={r}) | Speeds: {left_speed:.2f}, {right_speed:.2f}")



        else:

            # Structured street-like scanning: LEFT -> RIGHT -> LEFT

            scan_left = 1.0

            scan_right = 2.0

            scan_final_left = 3.0

            if search_step == 0:

                search_step = 1

                search_start_time = current_time

                print("Ball lost: scanning LEFT")

                left_speed, right_speed = -1.5, 1.5


            elif search_step == 1 and current_time - search_start_time < scan_left:


                left_speed, right_speed = -1.5, 1.5


            elif search_step == 1:

                search_step = 2

                search_start_time = current_time

                print("Scanning RIGHT")

                left_speed, right_speed = 1.5, -1.5


            elif search_step == 2 and current_time - search_start_time < scan_right:

                left_speed, right_speed = 0.5, -0.5


            elif search_step == 2:

                search_step = 3

                search_start_time = current_time

                print("Final look LEFT")

                left_speed, right_speed = -0.5, 0.5


            elif search_step == 3 and current_time - search_start_time < scan_final_left:

                left_speed, right_speed = -0.5, 0.5


            elif search_step == 3:

                search_step = 4

                print("Ball not found — full slow rotation")

                left_speed, right_speed = 0.5, -0.5


            elif search_step == 4:

                left_speed, right_speed = 0.5, -0.5

        # Clamp speeds
        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))
        robot.set_speed(left_speed, right_speed)

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
