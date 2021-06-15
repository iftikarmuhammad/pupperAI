r"""Running a pre-trained ppo agent on rex environments"""
import os
import site
import time

import tensorflow as tf
import numpy as np
from rex_gym.agents.scripts import utility
from rex_gym.agents.ppo import simple_ppo_agent
from rex_gym.util import action_mapper


class PolicyPlayer:
    def __init__(self, env_id: str, args: dict):
        self.gym_dir_path = '/home/ikar/.local/lib/python3.6/site-packages'
        self.env_id = env_id
        self.args = args

    def play(self):
        policy_dir = os.path.join(self.gym_dir_path, action_mapper.ENV_ID_TO_POLICY[self.env_id][0])
        print(policy_dir)
        config = utility.load_config(policy_dir)
        print(config)
        policy_layers = config.policy_layers
        value_layers = config.value_layers
        env = config.env(render=True, **self.args)
        network = config.network
        checkpoint = os.path.join(policy_dir, action_mapper.ENV_ID_TO_POLICY[self.env_id][1])
        np_path = os.path.join(self.gym_dir_path, 'rex_gym/rex_mix_2m_9.npy')
        action_out_path = os.path.join(self.gym_dir_path, 'rex_gym/pc_mix_2m_9_action.npy')
        dummy_obs = np.array([])
        action_out = [np.zeros(12)]
        with tf.Session() as sess:
            agent = simple_ppo_agent.SimplePPOPolicy(sess,
                                                     env,
                                                     network,
                                                     policy_layers=policy_layers,
                                                     value_layers=value_layers,
                                                     checkpoint=checkpoint)
            sum_reward = 0
            observation = env.reset()
            start = time.time()
            dummy_obs = [observation]
            while True:
                print(str(time.time() - start))
                if time.time() - start >= 60 and not(os.path.exists(np_path)):
                    with open(np_path, 'wb') as f:
                        np.save(f, dummy_obs)
                dummy_obs = np.concatenate((dummy_obs, [observation]), axis=0)
                action = agent.get_action([observation])
                observation, reward, done, _ = env.step(action[0])
                time.sleep(0.002)
                sum_reward += reward
                if done:
                    break
        # with tf.Session() as sess:
        #     agent = simple_ppo_agent.SimplePPOPolicy(sess,
        #                                              env,
        #                                              network,
        #                                              policy_layers=policy_layers,
        #                                              value_layers=value_layers,
        #                                              checkpoint=checkpoint)
        #     sum_reward = 0
        #     i = 0
        #     observation = env.reset()
        #     start = time.time()
        #     with open(np_path, 'rb') as f:
        #         dummy_obs = np.load(f)
        #     while True:
        #         # print(str(time.time() - start))
        #         # print('OBS: ' + str(type(dummy_obs[i][0])))
        #         action = agent.get_action([dummy_obs[i]])
        #         # print(action)
        #         # print('ACT: ' + str(type(action[0][0])))
        #         action_out = np.concatenate((action_out, action), axis=0)
        #         observation, reward, done, _ = env.step(action[0])
        #         time.sleep(0.002)
        #         i += 1
        #         sum_reward += reward
        #         if i == 3000:
        #             with open(action_out_path, 'wb') as f:
        #                 np.save(f, action_out)
        #         if done:
        #             break
