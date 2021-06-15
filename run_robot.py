r"""Running a pre-trained ppo agent on rex environments"""
import os
import site
import time
import math

import tensorflow as tf
import numpy as np

from rpi import (
	ServoParams,
	PWMParams,
	send_servo_commands,
	initialize_pwm,
	simple_ppo_agent,
	utility
)

TIME_STEP = 0.001
OBSERVATION_EPS = 0.01
STEP_PERIOD = 1.0 / 4.5
STEP_COUNTER = 0

INIT_POSE =         # 'stand_low': np.array([ # forward final
        #     0.15192765, -0.7552236, 1.88472,
        #     -0.15192765, -0.7552236, 1.88472,
        #     0.15192765, -0.7552236, 1.88472,
        #     -0.15192765, -0.7552236, 1.88472
        # ]),
        np.array([
            0.15, -0.74, 1.8,
            -0.15, -0.74, 1.8,
            0.15, -0.74, 1.8,
            -0.15, -0.74, 1.8
        ]),

ENV_ID_TO_POLICY = {
    'gallop': ('rex_gym/policies/galloping/balanced', 'model.ckpt-20000000'),
    'walk': ('pupper_rpi/policies/walking/alternating_legs/pupper-mix-2m-8/', 'model.ckpt-4000000'),
    'standup': ('rex_gym/policies/standup/pupper-500k/', 'model.ckpt-1000000'),
    'turn': ('rex_gym/policies/turn', 'model.ckpt-16000000')
}

# def _get_observation(self):
#     observation = []
#     roll, pitch, _ = self.rex.GetBaseRollPitchYaw()
#     roll_rate, pitch_rate, _ = self.rex.GetBaseRollPitchYawRate()
#     observation.extend([roll, pitch, roll_rate, pitch_rate])
#     observation[1] -= self.desired_pitch  # observation[1] is the pitch
#     self._observation = np.array(observation)
#     return self._observation

def get_observation():
	'get observation from the sensor: ROLL, PITCH, ROLL_RATE, PITCH_RATE'
	observation = np.zeros(4)
	return observation

def get_observation_dimension():
    return len(get_observation())

def get_observation_upper_bound():
    upper_bound = np.zeros(get_observation_dimension())
    upper_bound[0:2] = 2 * math.pi  # Roll, pitch, yaw of the base.
    upper_bound[2:4] = 2 * math.pi / TIME_STEP  # Roll, pitch, yaw rate.
    return upper_bound

def get_observation_lower_bound():
    lower_bound = -get_observation_upper_bound()
    return lower_bound

def observation_bound():
	observation_high = (get_observation_upper_bound() + OBSERVATION_EPS)
    observation_low = (get_observation_lower_bound() - OBSERVATION_EPS)
    return observation_low, observation_high

def action_bound():
	action_dim = 12
	action_high = np.array([0.25] * action_dim)
	action_low = -action_high
	return action_low, action_high

def reset():
	'reset all variables to initial state: ROLL, PITCH, ROLL_RATE, PITCH_RATE'
	STEP_COUNTER = 0
	return get_observation()

def convert_from_leg_model(action):
    motor_pose = np.zeros(NUM_MOTORS)
    for i in range(NUM_LEGS):
        if i % 2 == 0:
            motor_pose[3 * i] = 0.15
        else:
            motor_pose[3 * i] = -0.15
        motor_pose[3 * i + 1] = action[3 * i + 1]
        motor_pose[3 * i + 2] = action[3 * i + 2]
	# motor_pose = np.array([0, action[0], action[1],
	# 					0, action[2], action[3],
	# 					0, action[2], action[3],
	# 					0, action[0], action[1]])
    return motor_pose

def signal(action, t):
	initial_pose = INIT_POSE
	period = STEP_PERIOD
	l_extension = 0.225 * math.cos(3 * math.pi / period * t)
	l_swing = -l_extension
	extension = 0.225 * math.cos(3 * math.pi / period * t)
	swing = -extension
	pose = np.array([0, l_extension, extension,
					0, l_swing, swing,
					0, l_swing, swing,
					0, l_extension, extension])
	ol_signal = initial_pose + pose
	mix_signal = ol_signal + convert_from_leg_model(action)
	return mix_signal

def reshape_motor_command(signal):
	joint_angles = np.array([[signal[3], signal[0], signal[9], signal[6]]
							[signal[4], signal[1], signal[10], signal[7]]
							[signal[5], signal[2], signal[11], signal[8]]])
	return joint_angles

def step(pi_board, pwm_params, servo_params, joint_angles):
	send_servo_commands(pi_board, pwm_params, servo_params, joint_angles)
	STEP_COUNTER += 1
	return get_observation()

def start_pigpiod():
	subprocess.Popen(["sudo pkill pigpiod"])
	subprocess.Popen(["sudo pigpiod"])

def main():
	# HW Interface Initialization
	start_pigpiod()
	pi_board = pigpio.pi()
	pwm_params = PWMParams()
	initialize_pwm(pi_board, pwm_params)

	# Model Checkpoint Initialization
	gym_dir_path = '/home/ikar/.local/lib/python3.6/site-packages'  # kindly change this path into desired directory
	policy_dir = os.path.join(gym_dir_path, ENV_ID_TO_POLICY['walk'][0])
	config = utility.load_config(policy_dir)
	policy_layers = config.policy_layers
	value_layers = config.value_layers
	network = config.network
	checkpoint = os.path.join(policy_dir, ENV_ID_TO_POLICY['walk'][1])
	dummy_obs_path = os.path.join(gym_dir_path, '/rex_gym/rex_mix_5m.npy')

	observation_space = observation_bound()
	action_space = action_bound()

	with tf.Session() as sess:
		agent = simple_ppo_agent.SimplePPOPolicy(sess,
												 observation_space,
												 action_space,
												 network,
												 policy_layers=policy_layers,
												 value_layers=value_layers,
												 checkpoint=checkpoint)
		i = 0
		observation = reset()
		start = time.time()
		with open(dummy_obs_path, 'rb') as f:
        	dummy_obs = np.load(f)
		while True:
			print(str(time.time() - start))
			action = agent.get_action([dummy_obs[i]])
			t = STEP_COUNTER * TIME_STEP
			mix_signal = signal(action[0], t)
			joint_angles = reshape_motor_command(mix_signal)
			observation = step(pi_board, pwm_params, servo_params, joint_angles)
			time.sleep(0.002)
			i += 1
