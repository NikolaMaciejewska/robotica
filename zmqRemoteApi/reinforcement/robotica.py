'''
robotica.py

Provides the communication between CoppeliaSim robotics simulator and
external Python applications via the ZeroMQ remote API.

Copyright (C) 2024 Javier de Lope

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import math
import random

import numpy as np
import cv2
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class Coppelia():

    def __init__(self):
        print('*** connecting to coppeliasim')
        try:
            client = RemoteAPIClient()
            self.sim = client.getObject('sim')
        except Exception as e:
            raise RuntimeError(f"Failed to connect to CoppeliaSim: {e}")

    def start_simulation(self):
        # print('*** saving environment')
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        # print('*** stopping simulation')
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        # print('*** restoring environment')
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)
        print('*** done')

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped

    def randomize_obstacles_positions(self, obstacles_parent_name, x_range, y_range):
        """
        Randomly place all child obstacles under `obstacles_parent_name` within given x and y ranges.
        z_fixed is the fixed height at which obstacles will be placed.

        Parameters:
            obstacles_parent_name (str): Name/path of the parent object containing obstacles
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            z_fixed (float): Fixed z position for all obstacles (default 0.0)
        """

        parent_handle = self.sim.getObject(obstacles_parent_name)
        if parent_handle == -1:
            raise RuntimeError(f"Parent object '{obstacles_parent_name}' not found in the scene.")

        # Get children handles
        children = self.sim.getObjectsInTree(parent_handle, self.sim.handle_all, 3)
        print(children)

        print(f"Randomizing position of {len(children)} obstacles under '{obstacles_parent_name}'")

        for child_handle in children:
            # Get current position
            current_pos = self.sim.getObjectPosition(child_handle, -1)

            # Generate random x,y within range but keep original z
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            z = current_pos[2]

            self.sim.setObjectPosition(child_handle, -1, [x, y, z])

            # Optional: randomize orientation around z-axis
            current_orient = self.sim.getObjectOrientation(child_handle, -1)
            random_yaw = random.uniform(-math.pi, math.pi)
            new_orient = [current_orient[0], current_orient[1], random_yaw]
            self.sim.setObjectOrientation(child_handle, -1, new_orient)

        print("Obstacle positions randomized.")


class P3DX():

    num_sonar = 16
    sonar_max = 1.0

    def __init__(self, sim, robot_id, use_camera=False, use_lidar=False):
        self.sim = sim
        self.robot_id = robot_id
        print('*** getting handles', robot_id)
        self.left_motor = self.sim.getObject(f'/{robot_id}/leftMotor')
        self.right_motor = self.sim.getObject(f'/{robot_id}/rightMotor')

        if self.left_motor == -1 or self.right_motor == -1:
            raise RuntimeError("Motor handles not found! Check robot path or object names in the scene.")

        self.sonar = []
        for i in range(self.num_sonar):
            self.sonar.append(self.sim.getObject(f'/{robot_id}/ultrasonicSensor[{i}]'))
        if use_camera:
            self.camera = self.sim.getObject(f'/{robot_id}/camera')
        if use_lidar:
            self.lidar = self.sim.getObject(f'/{robot_id}/lidar')

        #NEW (store initial position)
        self.robot_handle = self.sim.getObject(f'/{robot_id}')
        self.initial_position = self.sim.getObjectPosition(self.robot_handle, -1)
        self.initial_orientation = self.sim.getObjectOrientation(self.robot_handle, -1)

    def get_sonar(self):
        readings = []
        for i in range(self.num_sonar):
            res,dist,_,_,_ = self.sim.readProximitySensor(self.sonar[i])
            readings.append(dist if res == 1 else self.sonar_max)
        return readings

    def get_image(self):
        if not hasattr(self, 'camera'):
            raise RuntimeError("Camera not initialized. Set `use_camera=True` in constructor.")

        # Updated API: returns [img_data, resolutionX, resolutionY]
        img_data, [resX, resY] = self.sim.getVisionSensorImg(self.camera)

        if img_data is None or resX == 0 or resY == 0:
            raise RuntimeError("Failed to retrieve image from vision sensor.")

        # Convert to NumPy array
        img = np.frombuffer(img_data, dtype=np.uint8).reshape((resY, resX, 3))

        # Flip image vertically (CoppeliaSim returns bottom-up)
        img = cv2.flip(img, 0)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def get_lidar(self):
        data = self.sim.getStringSignal('PioneerP3dxLidarData')
        if data is None:
            return []
        else:
            return self.sim.unpackFloatTable(data)

    def get_orientation(self):
        orient = self.sim.getObjectOrientation(self.robot_handle, -1)
        return orient

    def set_speed(self, left_speed, right_speed):
        current_handle = self.sim.getObject(f'/{self.robot_id}/leftMotor')
        #print(f"Stored left_motor handle: {self.left_motor}")
        #print(f"Current left_motor handle from sim: {current_handle}")
        if self.left_motor != current_handle:
            print("Warning: left_motor handle has changed. Possible object reload or scene reset?")
        if self.sim.getSimulationState() == self.sim.simulation_stopped:
            print("Warning: Trying to set motor speed while simulation is stopped.")
        else:
            self.sim.setJointTargetVelocity(self.left_motor, left_speed)
            self.sim.setJointTargetVelocity(self.right_motor, right_speed)

    #NEW
    def reset_to_initial_position(self):
        # Set position to initial
        self.sim.setObjectPosition(self.robot_handle, -1, self.initial_position)

        # Generate random yaw (rotation around z-axis)
        random_yaw = random.uniform(-math.pi, math.pi)  # from -180° to 180°
        random_orientation = list(self.initial_orientation)  # copy existing orientation
        random_orientation[2] = random_yaw  # modify yaw (z-axis angle)

        # Set the random orientation
        self.sim.setObjectOrientation(self.robot_handle, -1, random_orientation)

        # Stop motors
        #self.set_speed(0.0, 0.0)



def main(args=None):
    coppelia = Coppelia()
    robot = P3DX(coppelia.sim, 'PioneerP3DX')
    robot.set_speed(+1.2, -1.2)
    coppelia.start_simulation()
    while (t := coppelia.sim.getSimulationTime()) < 3:
        print(f'Simulation time: {t:.3f} [s]')
    coppelia.stop_simulation()


if __name__ == '__main__':
    main()
