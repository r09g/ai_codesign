import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# directory = 'outputs'
# energy = []
# latency = []
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     if os.path.isfile(f):
#         with open(f) as json_file:
#             data = json.load(json_file)
#             energy.append(data['energy'])
#             latency.append(data['latency'])
            

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(energy, latency, [a*b for a,b in zip(energy,latency)])
# ax.set_xlabel('Energy')
# ax.set_ylabel('Delay')
# ax.set_zlabel('EDP')
# ax.view_init(30, 135)
# plt.savefig('edp.png')


# plt.scatter(energy, latency)
# plt.xlabel('energy')
# plt.ylabel('latency')

# plt.savefig('energy-latency.png')

# df = pd.read_csv('log.csv', index_col=0)
# df['energy'] = energy
# df['latency'] = latency
# df['edp'] = [a*b for a,b in zip(energy,latency)]

# df.to_csv('data.csv')

name = ['TPU', 'Tesla_NPU', 'Meta_Prototype', 'Edge_TPU', 'Ascend']
energy = [688633757.6850001, 643201648.289, 660008431.645, 664688767.51, 644623518.8299999]
latency = [2061864, 1637957, 1468534, 1340280, 1772122]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for idx in range(len(energy)):
    ax.scatter(energy[idx], latency[idx], energy[idx]*latency[idx])
    ax.legend(name[idx])

ax.set_xlabel('Energy')
ax.set_ylabel('Delay')
ax.set_zlabel('EDP')
ax.view_init(30, 195)
plt.savefig('edp.png')
























