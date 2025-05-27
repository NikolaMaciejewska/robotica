import robotica
import numpy as np
import time
import random

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp  #proportional
        self.ki = ki  #integral
        self.kd = kd  #derivative

        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        #calculate the integral and derivative
        self.integral += error
        derivative = error - self.prev_error

        #PID
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        #save the current error for the next iteration
        self.prev_error = error

        return output

def avoid(readings):
    # include front-center and slight side sensors
    front_left = readings[2]
    front_center_left = readings[3]
    front_center_right = readings[4]
    front_right = readings[5]

    left_side = readings[1]
    right_side = readings[6]

    # more complete front detection
    front_min = min(front_left, front_center_left, front_center_right, front_right)

    if front_min < 0.25:
        if right_side > left_side:
            return +0.1, -0.8  # turn right
        else:
            return -0.8, +0.1  # turn left

    elif left_side < 0.15:
        return +1.3, +0.6

    elif right_side < 0.15:
        return +0.6, +1.3

    else:
        return None

def wander():
    base_speed = 5.0
    delta = random.uniform(-0.4, 0.4)  #small random turn
    return base_speed + delta, base_speed - delta

def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')

    right_pid = PID(kp=5.5, ki=0.0, kd=0.2)
    side_threshold = 0.5

    coppelia.start_simulation()

    while coppelia.is_running():
        readings = robot.get_sonar()

        #avoiding obstacles
        obstacle = avoid(readings)
        if obstacle:
            lspeed, rspeed = obstacle

        #wall following if close to the wall
        elif readings[5] < 1.0:  #right side
        #else:
            right_dist = readings[5]
            error = side_threshold - right_dist
            correction = right_pid.compute(error)

            base_speed = 5.0
            lspeed = np.clip(base_speed - correction, -1.0, 1.5)
            rspeed = np.clip(base_speed + correction, -1.0, 1.5)

        #wandering if far from the wall for more exploration (if we want just wall following then delete this part)
        else:
            lspeed, rspeed = wander()

        robot.set_speed(lspeed, rspeed)
        time.sleep(0.01)

    coppelia.stop_simulation()



if __name__ == '__main__':
    main()
