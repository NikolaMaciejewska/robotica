#DISCLAIMER: IT DOESN'T WORK WELL YET!!!

import robotica
import time
import cv2
import numpy as np

# PID parameters
Kp = 0.007
Ki = 0.0005
Kd = 0.003

base_speed = 0.8
max_speed = 2.0
min_speed = -0.5

ball_lost_timeout = 2.0  # seconds


def find_ball(img):
    blurred = cv2.GaussianBlur(img, (9, 9), 2)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=20, minRadius=5, maxRadius=100
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0]  # x, y, r
    return None


def pid_control(error, last_error, integral, dt):
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral


def avoid_obstacles(readings):
    front = min(readings[3], readings[4])
    left = readings[1]
    right = readings[6]

    if front < 0.25 or left < 0.1 or right < 0.1:
        return base_speed + 1.5, base_speed - 1.5
    return None


def main():
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    coppelia.start_simulation()

    last_error = 0.0
    integral = 0.0
    last_time = time.time()
    last_seen_time = time.time()

    while coppelia.is_running():
        left_speed = right_speed = 0.0

        readings = robot.get_sonar()
        img = robot.get_image()
        ball = find_ball(img)

        if ball is not None:
            x, y, r = map(int, ball)

            if r > 3:
                last_seen_time = time.time()

                img_center = img.shape[1] // 2
                error = img_center - x

                current_time = time.time()
                dt = current_time - last_time
                dt = max(dt, 0.05)
                last_time = current_time

                correction, integral = pid_control(error, last_error, integral, dt)
                last_error = error

                speed = base_speed * 0.5 if r > 40 else base_speed
                left_speed = speed + correction
                right_speed = speed - correction

                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        obstacle = avoid_obstacles(readings)
        if obstacle:
            left_speed, right_speed = obstacle
        elif ball is None:
            if time.time() - last_seen_time < ball_lost_timeout:
                left_speed = 0.5
                right_speed = -0.5
            else:
                left_speed = 0.0
                right_speed = 0.0

        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))

        robot.set_speed(left_speed, right_speed)

        cv2.imshow("Camera", img)
        cv2.waitKey(1)
        time.sleep(0.05)

    coppelia.stop_simulation()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
