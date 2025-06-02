import robotica
import time

# PID controller parameters
Kp = 0.9  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.3  # Derivative gain

target_distance = 0.3
max_speed = 1.5
min_speed = -0.5
base_speed = 1.3


def pid_wall_follow(error, last_error, integral, dt):
    # PID formula
    integral += error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0

    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

def avoid(readings):
    front_left = readings[3]
    front_right = readings[4]
    right = readings[6]
    left = readings[1]

    min_front = min(front_left, front_right)

    # If very close to an object, turn more sharply
    if min_front < 0.3 or left < 0.1 or right < 0.1:
        steer_strength = 2.0  # strong turn
    elif min_front < 0.45:
        steer_strength = 1.0  # gentle curve
    else:
        return None  # no avoidance needed

    left_speed = base_speed + steer_strength
    right_speed = base_speed - steer_strength

    return left_speed, right_speed

def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')

    coppelia.start_simulation()

    last_error = 0.0
    integral = 0.0
    last_time = time.time()
    wall_found = False

    while coppelia.is_running():
        readings = robot.get_sonar()
        obstacle = avoid(readings)

        #find wall
        if not wall_found:
            if min(readings[1], readings[2], readings[3], readings[4], readings[5]) < target_distance:
                wall_found = True
                continue
            left_speed, right_speed = base_speed, base_speed

        #avoid obstacle
        elif obstacle:
            left_speed, right_speed = obstacle

        else:
            side_distance = readings[1]

            # Compute the error
            error = target_distance - side_distance

            # Timing
            current_time = time.time()
            dt = current_time - last_time
            print(dt)
            dt = max(dt, 0.1)
            last_time = current_time

            # PID output
            correction, integral = pid_wall_follow(error, last_error, integral, dt)
            last_error = error

            left_speed = base_speed + correction
            right_speed = base_speed - correction

        # Clamp speeds
        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))

        robot.set_speed(left_speed, right_speed)

        time.sleep(0.05)

    coppelia.stop_simulation()


if __name__ == '__main__':
    main()