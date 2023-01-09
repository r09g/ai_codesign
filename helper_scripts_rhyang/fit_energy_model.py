import pickle
import math
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def mem_energy_expmodel(x, rf_w_size_exp, rf_w_bw_exp, rf_w_access, rf_o_size_exp, rf_o_bw_exp, rf_o_access, sram_size_exp, sram_bw_exp, sram_access, dram_size_exp, dram_bw_exp, dram_access):
    '''
        x[0]: rf_w_size
        x[1]: rf_w_bw
        x[2]: rf_o_size
        x[3]: rf_o_bw
        x[4]: sram_size
        x[5]: sram_bw
        x[6]: dram_size
        x[7]: dram_bw
    '''
    return rf_w_access*((x[0])**rf_w_size_exp)*((x[1])**rf_w_bw_exp) + rf_o_access*((x[2])**rf_o_size_exp)*((x[3])**rf_o_bw_exp) + sram_access*((x[4])**sram_size_exp)*((x[5])**sram_bw_exp) + dram_access*((x[6])**dram_size_exp)*((x[7])**dram_bw_exp)

def mem_energy_logmodel(x, rf_w_size_exp, rf_w_bw_exp, rf_w_access, rf_o_size_exp, rf_o_bw_exp, rf_o_access, sram_size_exp, sram_bw_exp, sram_access, dram_size_exp, dram_bw_exp, dram_access):
    '''
        x[0]: rf_w_size
        x[1]: rf_w_bw
        x[2]: rf_o_size
        x[3]: rf_o_bw
        x[4]: sram_size
        x[5]: sram_bw
        x[6]: dram_size
        x[7]: dram_bw
    '''
    return rf_w_access*(rf_w_size_exp*np.log(x[0]))*(rf_w_bw_exp*np.log(x[1])) + rf_o_access*(rf_o_size_exp*np.log(x[2]))*(rf_o_bw_exp*np.log(x[3])) + sram_access*(sram_size_exp*np.log(x[4]))*(sram_bw_exp*np.log(x[5])) + dram_access*(dram_size_exp*np.log(x[6]))*(dram_bw_exp*np.log(x[7]))



with open('design_data_point_fitmodel.pkl', 'rb') as file:
    design_data = pickle.load(file)

rf_w_size = []
rf_w_bw = []
rf_o_size = []
rf_o_bw = []
sram_size = []
sram_bw = []
dram_size = []
dram_bw = []
energy = []
for l in design_data:
    rf_w_size.append(l[0])
    rf_w_bw.append(l[1])
    rf_o_size.append(l[2])
    rf_o_bw.append(l[3])
    sram_size.append(l[4])
    sram_bw.append(l[5])
    dram_size.append(10000000000/8)
    dram_bw.append(64)
    energy.append(l[6])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(sram_size, sram_bw, energy)
# ax.set_xlabel('SRAM_Size (bytes)')
# ax.set_ylabel('SRAM_BW (bits)')
# ax.set_zlabel('Energy (nJ)')
# ax.view_init(30, 190)
# plt.savefig('sram_size-sram_bw-energy.png')

# warnings.filterwarnings('ignore')

plt.scatter(rf_w_size, energy)
# popt, pcov = curve_fit(mem_energy_logmodel, [rf_w_size, rf_w_bw, rf_o_size, rf_o_bw, sram_size, sram_bw, dram_size, dram_bw], energy)

# plt.plot(rf_w_size, mem_energy_logmodel([rf_w_size, rf_w_bw, rf_o_size, rf_o_bw, sram_size, sram_bw, dram_size, dram_bw], *popt))
plt.savefig('curve.png')
































