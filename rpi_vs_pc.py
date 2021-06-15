import numpy as np
import matplotlib.pyplot as plt

# path1 = 'rpi_tf15_mix_5m_action_4.npy'
path1 = 'pc_mix_2m_9_action.npy'
path2 = 'rpi_tf15_mix_2m_9_action.npy'

with open(path1, 'rb') as f:
	act1 = np.load(f)
with open(path2, 'rb') as f:
	act2 = np.load(f)

mae = np.mean(np.absolute(act1 - act2), axis=0)
mae_tot = np.mean(mae)
print(mae)
print(mae_tot)

plt.plot(act1[:,4], label='PC')
plt.plot(act2[:,4], label='RPi')
plt.legend()
plt.title('Joint Angle PCvsRPi no. 5')
plt.xlabel('Steps')
plt.ylabel('Joint Angle Values')
plt.show()

plt.plot(act1[:,0], label='PC')
plt.plot(act2[:,0], label='RPi')
plt.legend()
plt.title('Joint Angle PCvsRPi no. 1')
plt.xlabel('Steps')
plt.ylabel('Joint Angle Values')
plt.show()

plt.plot(np.absolute(act2 - act1)[:,0])
plt.title('Delta Joint Angle RPi float64vs32 no.1')
plt.xlabel('Steps')
plt.ylabel('Joint Angle Values')
plt.show()

plt.plot(np.absolute(act2 - act1)[:,4])
plt.title('Delta Joint Angle RPi float64vs32 no.5')
plt.xlabel('Steps')
plt.ylabel('Joint Angle Values')
plt.show()

index = []
for e in range(1,13):
	index.append(str(e))

plt.bar(index, mae)
plt.title('MAE for 12 Joint Angles - After Float Precision Fix')
plt.xlabel('Joint Angle Index')
plt.ylabel('MAE')
plt.show()
