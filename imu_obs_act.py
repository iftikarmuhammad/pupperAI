import numpy as np
import matplotlib.pyplot as plt


path1 = 'pupper_rpi_mix_2m_9_action_imu.npy'
path2 = 'pupper_mix_2m_9_imu.npy'

with open(path1, 'rb') as f:
	act1 = np.load(f)

for i in range(0,12):
	plt.plot(act1[:,i], label='JA no. ' + str(i))
plt.legend()
plt.title('Joint Angles (JA)')
plt.xlabel('Steps')
plt.ylabel('Joint Angle Values')
plt.show()

with open(path2, 'rb') as f:
	act2 = np.load(f)

plt.plot(act2[:,0], label='Roll')
plt.plot(act2[:,1], label='Pitch')
plt.legend()
plt.title('IMU Observation (Vector Euler Angle)')
plt.xlabel('Steps')
plt.ylabel('Vector Angle (rad)')
plt.show()

plt.plot(act2[:,2], label='Roll Rate')
plt.plot(act2[:,3], label='Pitch Rate')
plt.legend()
plt.title('IMU Observation (Gyro)')
plt.xlabel('Steps')
plt.ylabel('Angular Vel (rad/s)')
plt.show()