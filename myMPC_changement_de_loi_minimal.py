import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
import bisect

N = 2 # nombre de goutteurs
t_max = 24
max_gouteur = 1 # nombre de goutteurs ouverts en même temps
I = [0.4, 0.7]

D = np.random.randint(low=1, high=2, size=N) # retard des n systèmes
D = [1 for i in range(N)]

k = np.sum(D)+N # taille du vecteur X

U = np.zeros((N, t_max+1))
X = np.matrix(np.zeros((k, t_max +1)))
Y = np.matrix(np.zeros((N, t_max +1)))
Y[:, :max(D)] = 0.7*np.ones((Y[:, :max(D)].shape))

A = np.matrix(np.zeros((k,k))) # matrice qui contient sur les blocs diagonaux les matrices A de chaque système
B = np.matrix(np.zeros((k,N))) # matrice qui contient sur les blocs diagonaux les matrices B de chaque système
C = np.zeros((N, k))
vect_a = np.random.random(N) # vecteur contenant 'a' pour chaque système
vect_b = np.random.random(N) # vecteur contenant 'b' pour chaque système

vect_a = [0.95, 0.90, 0.92, 0.97]
vect_b = [0.15, 0.10, 0.12, 0.12]
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

def remplir_mat_B(mat_B):
    somme_d = 0
    current_p = 0
    for d in D:
        mat_B[somme_d, current_p] = 1
        somme_d += d+1
        current_p += 1
    return mat_B

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

def remplir_X0(M, Y):
    c_d = 0
    index_d = 0
    for d in D:
        c_d += d
        M[c_d] = Y[index_d, 0]
        c_d += 1
        index_d +=1
    return M

def remove(x, L):
    L1 = []
    for element in L:
        if not np.array_equal(element, x):
            L1.append(element)
    return L1

def détermine_all_u(Lu, X, i):
    if i == len(X):
        Lu.append(X)
    else:
        Y = copy.deepcopy(X)
        X[i] = 0
        return(détermine_all_u(Lu, X, i+1), détermine_all_u(Lu, Y, i+1))

def détermine_possible_u(Lu, max_gouteur):
    for u in Lu:
        compteur = 0
        for y in u:
            if y == 1:
                compteur += 1
        if compteur > max_gouteur:
            Lu = remove(u, Lu)
    return Lu
  
def get_last_index(L, x):
    last_index = 0
    for i in range(len(L)):
        if L[i] == x:
            last_index = i
    return last_index

def détermine_index_next_u(Lu):
    L_next_u = [[] for i in range(len(Lu))]
    for i1 in range(len(Lu)):
        L_hamming = []
        for i2 in range(len(Lu)):
            x = hamming(Lu[i1], Lu[i2])
            bisect.insort(L_hamming, x)
            index = get_last_index(L_hamming, x)
            L_next_u[i1] = L_next_u[i1][:index] + [i2] + L_next_u[i1][index:]
    return L_next_u

def make_u_plotable(U, i):
    for j in range(len(U)):
        if U[j] == 0:
            U[j] += 0.009*(i+1)
        else:
            U[j] -= 0.009*(i+1)

def get_position_in_L(L, e):
    for i in range(len(L)):
        if L[i] == e:
            return i

def calcul_u_optimal(Lu, X, Y, A, B, C, t_min, t_max, I, N, D):
    max_i_u = len(Lu)-1
    pos_i_u = 0
    i_um = 0
    c_t = 0
    L_iu_ium= [[0, 0] for i in range(t_max+1)]
    L_u = []
    is_not_finished = True
    L_index_next_u = détermine_index_next_u(Lu)
    while is_not_finished and c_t < t_max:
        c_u = Lu[L_index_next_u[i_um][pos_i_u]]
        X[:, c_t + 1 ] = A*X[:, c_t] + B*np.matrix(c_u).T
        Y[:, c_t + 1] = C*X[:, c_t + 1]
        compteur_système = 0
        for n in range(N):
            if I[0] < Y[n, c_t + 1] < I[1]:
                compteur_système += 1
                if compteur_système == N:
                    L_u.append(c_u)
                    c_t += 1
                    i_u = L_index_next_u[i_um][pos_i_u]
                    pos = get_position_in_L(L_index_next_u[i_u], L_index_next_u[i_um][pos_i_u])
                    i_um = i_u
                    pos_i_u = pos 
                    L_iu_ium[c_t] = [i_um, pos_i_u]   
            else:
                t = c_t - D[n]
                while get_position_in_L(L_index_next_u[L_iu_ium[t][0]], L_iu_ium[t][1]) == max_i_u and t >= 1:
                    t = t - 1
                if get_position_in_L(L_index_next_u[L_iu_ium[t][0]], L_iu_ium[t][1]) == max_i_u:
                    is_not_finished = False
                c_t = t
                c_ium = L_iu_ium[t][0]
                c_pos_iu  = L_iu_ium[t][1]
                i_um = c_ium
                pos_i_u = c_pos_iu +1  
                L_iu_ium[t] = [i_um, pos_i_u]
                L_u = L_u[:t]
                break
    if len(L_u) < t_max:
        print("NO SOLUTION FOUND")
    mat_action_min_cout = np.zeros((N, t_max+1))
    for i in range(len(L_u)):
        mat_action_min_cout[:, i] = L_u[i]
    return mat_action_min_cout


def prédiction(A, B, C, X, Y, I):
    Lu = []
    détermine_all_u(Lu, np.ones(N), 0)
    Lu = détermine_possible_u(Lu, max_gouteur)
    U[:, :] = calcul_u_optimal(Lu, X, Y, A, B, C, max(D), t_max, I, N, D)
    for t in range(1, t_max):
        X[:, t] = A*X[:, t-1] + B*np.matrix(U[:, t-1]).T
        Y[:, t] = C*X[:,  t-1]
    

A = remplir_mat_A(A)
B = remplir_mat_B(B)
C = remplir_mat_C(C)
X[:, :max(D)] = remplir_X0(X[:, :max(D)], Y)


prédiction(A, B, C, X, Y, I )

plt.figure(figsize=(12,7))
plt.subplot(122)
for i in range(len(U)):
    make_u_plotable(U[i, :], i)
    plt.scatter([j for j in range(t_max + 1)], U.T[:, i],  label=f"système {i+1}")
plt.text(4, 0.68, f"number of systems = {N}")
plt.text(4, 0.64, f"max dripper = {max_gouteur}")

for i in range(N):
    plt.text(0.5, 0.50-i*0.04, f"a{i+1}={vect_a[i]}")
    plt.text(7, 0.50-i*0.04, f"b{i+1}={vect_b[i]}")

plt.xlabel("Time")
plt.title("U en fonction de k")
plt.legend()

plt.subplot(121)
for i in range(len(Y)):
    plt.plot(Y.T[:, i], label=f"système{i+1}")
plt.grid()
plt.xticks([i for i in range(t_max+1)])
plt.title("Y en fonction de k")
plt.xlabel("Time")

plt.legend()
plt.show()
