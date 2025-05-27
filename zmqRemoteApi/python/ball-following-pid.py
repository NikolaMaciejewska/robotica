import robotica
import time
import cv2
import numpy as np

# PID parameters for ball tracking
Kp = 0.007
Ki = 0.0005
Kd = 0.003

base_speed = 3.0
max_speed = 1.5
min_speed = -0.5

ball_lost_timeout = 2.0

def find_red_ball(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # red color range (wraps around HSV 0)
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
        return tuple(map(int, circles[0][0]))  # (x, y, radius)

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
    last_seen_time = 0.0

    while coppelia.is_running():
        readings = robot.get_sonar()
        img = robot.get_image()
        ball = find_red_ball(img)

        current_time = time.time()
        dt = current_time - last_time
        dt = max(dt, 0.05)
        last_time = current_time

        obstacle = avoid_obstacles(readings)
        left_speed, right_speed = 0.0, 0.0

        if ball is not None:
            x, y, r = map(int, ball)
            last_seen_time = time.time()

            img_center = img.shape[1] // 2
            error = img_center - x

            correction, integral = pid_control(error, last_error, integral, dt)
            correction = max(-base_speed, min(base_speed, correction))
            last_error = error

            speed = base_speed * 0.5 if r > 40 else base_speed

            left_speed = speed - correction
            right_speed = speed + correction

            print(f"Ball detected: x={x}, radius={r}, error={error}, speeds=({left_speed:.2f}, {right_speed:.2f})")

        if obstacle:
            left_speed, right_speed = obstacle
            print("Obstacle detected - turning")

        elif ball is None:
            left_speed = 0.5
            right_speed = -0.5
            print("Searching for ball - spinning")

        # Clamp speeds
        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))

        robot.set_speed(left_speed, right_speed)
        print(f"Set speeds => Left: {left_speed:.2f}, Right: {right_speed:.2f}")

        # Visualize
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
