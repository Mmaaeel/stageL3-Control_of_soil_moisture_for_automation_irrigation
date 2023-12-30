import numpy as np
import control


import matplotlib.pyplot as plt
N = 2
t_max = 100

D = np.random.randint(low=1, high=5, size=N) # retard des n systèmes
D = np.ones(N, int)
k = np.sum(D)+N # taille du vecteur X
X = np.matrix(np.zeros((k, t_max)))
X_ref = np.matrix(np.zeros((k, 1)))

Y_ref = 0.5 * np.ones((N,1))
U = np.zeros((N, t_max))
Y = np.zeros((N, t_max))
vect_a = np.random.random(N) # vecteur contenant 'a' pour chaque système
vect_b = np.random.random(N) # vecteur contenant 'b' pour chaque système


vect_a = [0.95, 0.90]
vect_b = [0.15, 0.10]

mat_A = np.matrix(np.zeros((k,k))) # matrice qui contient sur les blocs diagonaux les matrices A de chaque système
mat_B = np.matrix(np.zeros((k,N))) # matrice qui contient sur les blocs diagonaux les matrices B de chaque système
mat_C = np.zeros((N, k))

Y[:, 0] = [0.7, 0.45]
Y_ref = np.zeros((2, 1))
Y_ref[:, 0] = [0.5, 0.6]

def remplir_mat_A():
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

def remplir_mat_B():
    somme_d = 0
    current_p = 0
    for d in D:
        mat_B[somme_d, current_p] = 1
        somme_d += d+1
        current_p += 1

def remplir_mat_C():
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
        
def remplir_X0(M, Y):
    c_d = 0
    index_d = 0
    for d in D:
        c_d += d
        M[c_d] = Y[index_d, 0]
        c_d += 1
        index_d +=1

def remplir_X():
    for t in range(1, t_max):
        X[:, t] = (mat_A - mat_B*K)*X[:, t-1] + X_c

        
def remplir_Y():
    Y[:,:] = np.matrix(mat_C)*np.matrix(X)




remplir_X0(X, Y)
remplir_X0(X_ref, Y_ref)
remplir_mat_A()
remplir_mat_B()
remplir_mat_C()
Q = np.identity(k)
R = np.identity(N)
K = control.dlqr(mat_A, mat_B, Q, R)[0]
X_c = (np.eye(k) - mat_A + mat_B*K)*X_ref

remplir_X()
remplir_Y()

for i in range(len(Y)):
    plt.plot(Y.T[:, i], label=f"systeme {i+1}")
plt.legend()
plt.title("Y in relation to time")
plt.xlabel("Time")
plt.ylabel("Soil moisture")
plt.show()
