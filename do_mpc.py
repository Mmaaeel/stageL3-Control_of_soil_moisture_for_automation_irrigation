import numpy as np
import do_mpc
from casadi import *
import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

N = 5
t_max = 1
water_capacity = 5.

D = np.random.randint(low=1, high=3, size=N) # retard des n systèmes
k = np.sum(D)+N # taille du vecteur X

vect_a = np.random.random(N) # vecteur contenant 'a' pour chaque système
vect_a = [0.5, 0.5, 0.5, 0.5, 0.5]
vect_b = np.random.random(N) # vecteur contenant 'b' pour chaque système
vect_b = [0.5, 0.5, 0.5, 0.5, 0.5]
def remplir_mat_A(mat_A):
    somme_d = 0
    index_d = 0
    for d in D:
        len_currentA = d+1
        for i in range(len_currentA):
            if 0 < i < len_currentA - 1:
                mat_A[somme_d + i, somme_d] = 1
            if i == len_currentA -1:
                mat_A[somme_d + i, somme_d + i -1] = vect_b[index_d]
                mat_A[somme_d + i, somme_d + i] = vect_a[index_d]
        somme_d += d+1
        index_d += 1
    return mat_A
A = remplir_mat_A(np.zeros((k,k)))

def remplir_mat_B(mat_B):
    somme_d = 0
    current_p = 0
    for d in D:
        mat_B[somme_d, current_p] = 1
        somme_d += d+1
        current_p += 1
    return mat_B
B = remplir_mat_B(np.zeros((k,N)))

def remplir_mat_C(mat_C):
    c_d = 0
    c_ligne = 0
    for i in range(len(D)):
        d = D[i]
        if i == 0:
            c_d += d
            mat_C[c_ligne, c_d] = 1
        if i > 0:
            c_d += d+1
            mat_C[c_ligne, c_d] = 1
        c_ligne += 1
    return mat_C
C = remplir_mat_C(np.zeros((N,k)))

def calcul_mterm(A):
    s = A.shape
    res = 0
    for i in range(s[0]):
        for j in range(s[1]):
            res += (A[i, j]*(A[i, j]-1))**2
            """if A[i, j] < 0:
                res += np.log(-A[i, j])"""
    return res

def calcul_somme_vect(U):
    res = 0
    for i in range(U.shape[0]):
        res += U[i]
    return res

def remplir_X0(M, val):
    c_d = 0
    for d in D:
        c_d += d
        M[c_d] = val
        c_d += 1
    return M


model_type = 'discrete' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

U = model.set_variable(var_type='_u', var_name='U', shape=(N, t_max))
SU = model.set_variable(var_type='_x', var_name='SU', shape=(1,1))
X = model.set_variable(var_type='_x', var_name='X', shape=(k, t_max))
Y = model.set_variable(var_type='_x', var_name='Y', shape=(N, t_max))

X_next = mtimes(A, X) + mtimes(B, U)
Yk = mtimes(C, X)
SU_next = calcul_somme_vect(U)

model.set_rhs("X", X_next)
model.set_rhs("Y", Yk)
model.set_rhs("SU", SU_next)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    't_step': 0.1,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

mterm = 0*calcul_mterm(U)
lterm = 10**3*calcul_mterm(U)

mpc.set_rterm(U=10**3)

mpc.set_objective(mterm=mterm*0, lterm=lterm*0)

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)

mpc.bounds['lower','_x', 'Y'] = 0.4
mpc.bounds['upper','_x', 'Y'] = 0.7
mpc.bounds['lower','_u', 'U'] = 0
"""mpc.bounds['upper','_u', 'U'] = 1"""
mpc.bounds['lower','_x', 'SU'] = 0
mpc.bounds['upper','_x', 'SU'] = water_capacity
# mpc.bounds['lower','_x', "VU"] = N

mpc.setup()
p_template = simulator.get_p_template()
simulator.setup()

x0 = np.zeros((N + k + 1, 1))
x0[1:k+1] = remplir_X0(x0[1:k+1], 0.5)
x0[k+1:N+k+1] = np.random.random((N,1))

simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(10,6))
fig.align_ylabels()
for g in [sim_graphics, mpc_graphics]:
    # g.add_line(var_type='_x', var_name='X', axis=ax[0])
    g.add_line(var_type='_x', var_name='Y', axis=ax[0])
    g.add_line(var_type="_u", var_name='U', axis=ax[1])

ax[0].set_ylabel('Y')
ax[1].set_ylabel('U')
ax[1].set_xlabel('time')

u0 = np.zeros((N*t_max, 1))
for i in range(20):
    simulator.make_step(u0)

"""sim_graphics.plot_results()
sim_graphics.reset_axes()
"""
blockPrint()
u0 = mpc.make_step(x0)

sim_graphics.clear()
# mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()
mpc_graphics.pred_lines['_x', 'Y']

simulator.reset_history()
mpc.reset_history()
simulator.x0 = x0

for i in range(20):
    u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)
enablePrint()
# Plot predictions from t=0
# mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()

plt.show()