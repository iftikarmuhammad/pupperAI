import collections
import copy
import math
import re
import pigpio
import numpy as np
from datetime import datetime
from . import motor
from ..util import pybullet_data
import pyrealsense2.pyrealsense2 as rs

from rex_gym.rpi.hardware_interface import send_servo_commands, initialize_pwm
from rex_gym.rpi.pupper_config import ServoParams, PWMParams

INIT_POSITION = [0, 0, 0.21]
INIT_RACK_POSITION = [0, 0, 1]
INIT_ORIENTATION = [0, 0, 0, 1]

class Rex(object):
    """The Rex class that simulates a quadruped robot."""
    half_pi = math.pi / 2.0
    knee_angle = -2.1834
    INIT_POSES = {
        'stand_low': np.array([
            0.1, 0.74, -1.06,
            -0.1, 0.74, -1.06,
            0.1, 0.74, -1.06,
            -0.1, 0.74, -1.06])
    }

    def __init__(self,
                 time_step=0.01,
                 action_repeat=1,
                 self_collision_enabled=False,
                 motor_velocity_limit=np.inf,
                 pd_control_enabled=False,
                 accurate_motor_model_enabled=False,
                 remove_default_joint_damping=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 pd_latency=0.0,
                 control_latency=0.0,
                 torque_control_enabled=False,
                 motor_overheat_protection=False,
                 on_rack=False,
                 pose_id='stand_low'):
        """Constructs a Rex and reset it to the initial states.

        Args:
          pybullet_client: The instance of BulletClient to manage different
            simulations.
          urdf_root: The path to the urdf folder.
          time_step: The time step of the simulation.
          action_repeat: The number of ApplyAction() for each control step.
          self_collision_enabled: Whether to enable self collision.
          motor_velocity_limit: The upper limit of the motor velocity.
          pd_control_enabled: Whether to use PD control for the motors.
          accurate_motor_model_enabled: Whether to use the accurate DC motor model.
          remove_default_joint_damping: Whether to remove the default joint damping.
          motor_kp: proportional gain for the accurate motor model.
          motor_kd: derivative gain for the accurate motor model.
          pd_latency: The latency of the observations (in seconds) used to calculate
            PD control. On the real hardware, it is the latency between the
            microcontroller and the motor controller.
          control_latency: The latency of the observations (in second) used to
            calculate action. On the real hardware, it is the latency from the motor
            controller, the microcontroller to the host (Nvidia TX2).
          observation_noise_stdev: The standard deviation of a Gaussian noise model
            for the sensor. It should be an array for separate sensors in the
            following order [motor_angle, motor_velocity, motor_torque,
            base_roll_pitch_yaw, base_angular_velocity]
          torque_control_enabled: Whether to use the torque control, if set to
            False, pose control will be used.
          motor_overheat_protection: Whether to shutdown the motor that has exerted
            large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
            (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in rex.py for more
            details.
          on_rack: Whether to place the Rex on rack. This is only used to debug
            the walking gait. In this mode, the Rex's base is hanged midair so
            that its walking gait is clearer to visualize.
        """
        self.num_motors = 12
        self.num_legs = 4
        self._action_repeat = action_repeat
        self._self_collision_enabled = self_collision_enabled
        self._motor_velocity_limit = motor_velocity_limit
        self._pd_control_enabled = pd_control_enabled
        self._motor_direction = [1 for _ in range(12)]
        self._observed_motor_torques = np.zeros(self.num_motors)
        self._applied_motor_torques = np.zeros(self.num_motors)
        self._max_force = 4
        self._pd_latency = pd_latency
        self._control_latency = control_latency
        self._accurate_motor_model_enabled = accurate_motor_model_enabled
        self._remove_default_joint_damping = remove_default_joint_damping
        self._observation_history = collections.deque(maxlen=100)
        self._control_observation = []
        self._chassis_link_ids = [-1]
        self._leg_link_ids = []
        self._motor_link_ids = []
        self._foot_link_ids = []
        self._torque_control_enabled = torque_control_enabled
        self._motor_overheat_protection = motor_overheat_protection
        self._on_rack = on_rack
        self._pose_id = pose_id
        # @TODO fix MotorModel
        if self._accurate_motor_model_enabled:
            self._kp = motor_kp
            self._kd = motor_kd
            self._motor_model = motor.MotorModel(torque_control_enabled=self._torque_control_enabled,
                                                 kp=self._kp,
                                                 kd=self._kd)
        elif self._pd_control_enabled:
            self._kp = 8
            self._kd = 0.3
        else:
            self._kp = 1
            self._kd = 1
        self.time_step = time_step
        self._step_counter = 0
        # reset_time=-1.0 means skipping the reset motion.
        # See Reset for more details.
        # self.GetPoseData()
        self.Reset(reset_time=-1)
        self.init_on_rack_position = INIT_RACK_POSITION
        self.init_position = INIT_POSITION
        self.initial_pose = self.INIT_POSES[pose_id]
        self._reset_time = datetime.now()

    def GetTimeSinceReset(self):
        # print('TE REX : ' + str(self.time_step))
        return self._step_counter * self.time_step

    def Step(self, action):
        # print('REX TS, AR : %s, %s' % (str(self.time_step), str(self._action_repeat)))
        self.GetPoseData()
        # for _ in range(self._action_repeat):
        self.ApplyAction(action)
        self.ReceiveObservation()
        self._step_counter += 1

    def Terminate(self):
        pass

    @staticmethod
    def IsObservationValid():
        """Whether the observation is valid for the current time step.

        In simulation, observations are always valid. In real hardware, it may not
        be valid from time to time when communication error happens.

        Returns:
          Whether the observation is valid for the current time step.
        """
        return True

    def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
        """Reset the Rex to its initial states.

        Args:
          reload_urdf: Whether to reload the urdf file. If not, Reset() just place
            the Rex back to its starting position.
          default_motor_angles: The default motor angles. If it is None, Rex
            will hold a default pose for 100 steps. In
            torque control mode, the phase of holding the default pose is skipped.
          reset_time: The duration (in seconds) to hold the default motor angles. If
            reset_time <= 0 or in torque control mode, the phase of holding the
            default pose is skipped.
        """
        print("reset")
        if reset_time > 0.0:
            episode_length = datetime.now() - self._reset_time
            print('EPISODE LENGTH : ' + str(episode_length.total_seconds()))
            self._reset_time = datetime.now()
        if self._on_rack:
            init_position = INIT_RACK_POSITION
        else:
            init_position = INIT_POSITION

        if reload_urdf:
            self.pi_board = pigpio.pi()
            self.servo_params = ServoParams()
            self.pwm_params = PWMParams()
            initialize_pwm(self.pi_board, self.pwm_params)

            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            self.pipe.start(cfg)
            self.ResetPose()
        else:
            self.pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.pose)
            self.pipe.start(cfg)
            self.ResetPose()
            input("Kindly put pupper on the initial position and press Enter...")
        self._step_counter = 0
        self.GetPoseData()
        self.ReceiveObservation()

    def reshape_motor_command(self, signal):
        joint_angles = np.array([[signal[3], signal[0], signal[9], signal[6]],
                                                        [signal[4], signal[1], signal[10], signal[7]],
                                                        [signal[5], signal[2], signal[11], signal[8]]])
        return joint_angles

    def ResetPose(self):
        """Reset the pose of the Rex.

        Args:
          add_constraint: Whether to add a constraint at the joints of two feet.
        """
        motor_commands = self.reshape_motor_command(self.INIT_POSES[self._pose_id])
        send_servo_commands(self.pi_board, self.pwm_params, self.servo_params, motor_commands)

    def GetPoseData(self):
        frames = self.pipe.wait_for_frames()
        pose = frames.get_pose_frame()
        if pose:
            self.data = pose.get_pose_data()

    def GetBasePosition(self):
        """Get the position of Rex's base.

        Returns:
          The position of Rex's base.
        """
        position = self.data.translation
        return np.asarray([-position.z, position.x, position.y])

    def GetBaseVelocity(self):
        """Get the position of Rex's base.

        Returns:
          The position of Rex's base.
        """
        velocity = self.data.velocity
        return velocity

    def GetBaseRollPitchYaw(self):
        """Get Rex's base orientation in euler angle in the world frame.

        Returns:
          A tuple (roll, pitch, yaw) of the base in world frame.
        """
        data = self.data
        w = data.rotation.w
        x = -data.rotation.z
        y = data.rotation.x
        z = -data.rotation.y

        pitch =  -math.asin(2.0 * (x*z - w*y))
        roll  =  math.atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z)
        yaw   =  math.atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z)
        return np.asarray([roll, pitch, yaw])

    def GetBaseOrientation(self):
        """Get the orientation of Rex's base, represented as quaternion.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          The orientation of Rex's base polluted by noise and latency.
        """
        data = self.data
        w = data.rotation.w
        x = -data.rotation.z
        y = data.rotation.x
        z = -data.rotation.y
        # print(np.asarray([x, y, z, w]))
        # return np.asarray([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
        return np.asarray([x, y, z, w])

    def GetBaseRollPitchYawRate(self):
        """Get the rate of orientation change of the Rex's base in euler angle.

        This function mimicks the noisy sensor reading and adds latency.
        Returns:
          rate of (roll, pitch, yaw) change of the Rex's base polluted by noise
          and latency.
        """
        rpy_rate = self.data.angular_velocity
        # return np.asarray([0.0, 0.0, 0.0])
        return np.asarray([rpy_rate.x, rpy_rate.y, rpy_rate.z])

    def GetActionDimension(self):
        """Get the length of the action list.

        Returns:
          The length of the action list.
        """
        return self.num_motors

    def ApplyMotorLimits(self, joint_angles):
        eps = 0.001
        for i in range(len(joint_angles)):
            LIM = MOTOR_LIMITS_BY_NAME[MOTOR_NAMES[i]]
            joint_angles[i] = np.clip(joint_angles[i], LIM[0] + eps,
                                      LIM[1] - eps)
        return joint_angles

    def ApplyAction(self, motor_commands):
        """Set the desired motor angles to the motors of the Rex.

        The desired motor angles are clipped based on the maximum allowed velocity.
        If the pd_control_enabled is True, a torque is calculated according to
        the difference between current and desired joint angle, as well as the joint
        velocity. This torque is exerted to the motor. For more information about
        PD control, please refer to: https://en.wikipedia.org/wiki/PID_controller.

        Args:
          motor_commands: The eight desired motor angles.
          motor_kps: Proportional gains for the motor model. If not provided, it
            uses the default kp of the Rex for all the motors.
          motor_kds: Derivative gains for the motor model. If not provided, it
            uses the default kd of the Rex for all the motors.
        """
        # motor_commands = self.ApplyMotorLimits(motor_commands)
        motor_commands = self.reshape_motor_command(motor_commands)
        send_servo_commands(self.pi_board, self.pwm_params, self.servo_params, motor_commands)

    def GetTrueObservation(self):
        observation = []
        # observation.extend(self.GetTrueMotorAngles())
        # observation.extend(self.GetTrueMotorVelocities())
        # observation.extend(self.GetTrueMotorTorques())
        observation.extend(self.GetBaseOrientation())
        observation.extend(self.GetBaseRollPitchYawRate())
        return observation

    def ReceiveObservation(self):
        """Receive the observation from sensors.

        This function is called once per step. The observations are only updated
        when this function is called.
        """
        self._observation_history.appendleft(self.GetTrueObservation())

    def SetTimeSteps(self, action_repeat, simulation_step):
        """Set the time steps of the control and simulation.

        Args:
          action_repeat: The number of simulation steps that the same action is
            repeated.
          simulation_step: The simulation time step.
        """
        self.time_step = simulation_step
        self._action_repeat = action_repeat

    @property
    def chassis_link_ids(self):
        return self._chassis_link_ids
