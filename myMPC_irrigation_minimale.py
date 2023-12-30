import numpy as np
import copy
import matplotlib.pyplot as plt
import bisect

N = 3 # nombre de goutteurs 
t_max = 24
max_gouteur = 1 # nombre de goutteurs ouverts en même temps
I = [0.4, 0.7]

D = np.random.randint(low=1,  high=3, size=N) # retard des n systèmes

k = np.sum(D)+N # taille du vecteur X


U = np.zeros((N, t_max))
X = np.matrix(np.zeros((k, t_max +1)))
Y = np.matrix(np.zeros((N, t_max +1)))
Y[:, :max(D)] = 0.7*np.ones((Y[:, :max(D)].shape))


A = np.matrix(np.zeros((k,k))) # matrice qui contient sur les blocs diagonaux les matrices A de chaque système
B = np.matrix(np.zeros((k,N))) # matrice qui contient sur les blocs diagonaux les matrices B de chaque système
C = np.zeros((N, k))
# vect_a = np.random.uniform(0.85, 0.95, N) # vecteur contenant 'a' pour chaque système
# vect_b = np.random.uniform(0.3, 0.4, N) # vecteur contenant 'b' pour chaque système

vect_a = [0.95, 0.9, 0.96]
vect_b = [0.15, 0.1, 0.15]
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

def cout_changement(mémoire_action):
    compteur = 0
    for i in range(len(mémoire_action) - 1):
        if mémoire_action[i+1] != mémoire_action[i]:
            compteur += 1
    return compteur

def remove(x, L):
    L1 = []
    for element in L:
        if not np.array_equal(element, x):
            L1.append(element)
    return L1

def get_last_index(L, x):
    last_index = 0
    for i in range(len(L)):
        if L[i] == x:
            last_index = i
    return last_index

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

def tri_Lu(Lu):
    Lu_tri = []
    L_poids = []
    for M in Lu:
        compteur = 0
        for e in M:
            if e == 1:
                compteur += 1
        bisect.insort(L_poids, compteur)
        index = get_last_index(L_poids, compteur)
        Lu_tri = Lu_tri[:index] + [M] + Lu_tri[index:] 
    return Lu_tri

def make_u_plotable(U, i):
    for j in range(len(U)):
        if U[j] == 0:
            U[j] += 0.009*(i+1)
        else:
            U[j] -= 0.009*(i+1)

def calcul_u_optimal(Lu, X, Y, A, B, C, t_min, t_max, I, N, D):
    index_c_u = 0
    c_t = 0
    index_memoire_action = [0 for i in range(t_min+1)]
    index_action_min_cout = []
    min_cout = 10**8
    is_not_finished = True
    Lu_tri = tri_Lu(Lu)
    while is_not_finished and c_t < t_max:
        c_u = Lu_tri[index_c_u]
        X[:, c_t + 1] = A*X[:, c_t] + B*np.matrix(c_u).T
        Y[:, c_t + 1] = C*X[:, c_t + 1]
        compteur_système = 0
        for z in range(N):
            if  I[0] < Y[:, c_t + 1][z] < I[1]:
                compteur_système += 1
                if compteur_système == N:
                    index_memoire_action.append(index_c_u)
                    index_c_u = 0
                    c_t += 1
            else:
                t = c_t - D[z] 
                while index_memoire_action[t] == len(Lu)-1 and t > t_min:
                    t = t - 1
                if t == t_min and index_memoire_action[t] == len(Lu)-1:
                    is_not_finished = False
                else:
                    c_t = t
                    index_c_u = index_memoire_action[t] + 1 
                    index_memoire_action = index_memoire_action[:t]
                break     
    if len(index_memoire_action) < t_max:
        return None
    else:
        mat_action_min_cout = np.zeros((N, t_max))
        for i in range(len(index_memoire_action)):
            mat_action_min_cout[:, i] = Lu[index_memoire_action[i]]
        return mat_action_min_cout


def prédiction(A, B, C, X, Y, I):
    Lu = []
    détermine_all_u(Lu, np.ones(N), 0)
    Lu = détermine_possible_u(Lu, max_gouteur)
    var = calcul_u_optimal(Lu, X, Y, A, B, C, max(D), t_max, I, N, D)
    if var is None:
        return "NO SOLUTION"
    else:
        U[:, :] = var
        for t in range(1, t_max):
            X[:, t] = A*X[:, t-1] + B*np.matrix(U[:, t-1]).T
            Y[:, t] = C*X[:,  t-1]
        
    

A = remplir_mat_A(A)
B = remplir_mat_B(B)
C = remplir_mat_C(C)
X[:, :max(D)] = remplir_X0(X[:, :max(D)], Y)
x = prédiction(A, B, C, X, Y, I )
if x == "NO SOLUTION":
    print("NO SOLUTION FOUND")
else:
    plt.figure(figsize=(12,7))
    plt.subplot(122)
    for i in range(len(U)):
        make_u_plotable(U[i, :], i)
        plt.scatter([j for j in range(t_max)], U.T[:, i],  label=f"système {i+1}")
    plt.text(4, 0.68, f"number of systems = {N}")
    plt.text(4, 0.64, f"max dripper = {max_gouteur}")
    plt.xlabel("Time")
    plt.title("U en fonction de k")
    plt.legend()

    for i in range(N):
        plt.text(0.5, 0.50-i*0.04, f"a{i+1}={vect_a[i]}")
        plt.text(7, 0.50-i*0.04, f"b{i+1}={vect_b[i]}")

    plt.subplot(121)
    for i in range(len(Y)):
        plt.plot(Y.T[:, i], label=f"système{i+1}")
    plt.grid()
    plt.xticks([i for i in range(t_max+1)])
    plt.title("Y en fonction de k")
    plt.xlabel("Time")
    plt.ylabel("Soil moisture")
    plt.legend()
    plt.show()
    