import numpy as np
import matplotlib.pyplot as plt
t_end=100
n=5
y=np.zeros([n,t_end])
u=np.zeros([n,t_end])
Ta=np.random.rand(n)*.006+.992
Td=np.random.randint(8,10,size=n)
Ty_int=np.random.uniform(0.5, 0.7, n)
t_spam=np.arange(0,t_end)
b=2.5
y[:,0]=Ty_int
y[:,-1]=Ty_int
def Hysteresis(u,y):
    y_m=.4
    y_M=.7
    if y<y_m and u==0:
        u_f=1
    elif y>y_M and u==1:
        u_f=0    
    else:
        u_f=u
    return u_f

    
for i in np.arange(0,n):
    d=Td[i]
    up=0
    for t in t_spam:
        yp0=y[i,t]   
        for tp in np.arange(0,d+1):
            yp1=Ta[i]*yp0+(1-Ta[i])*b*up
            yp0=yp1

        up=Hysteresis(up,yp1)
        u[i,t]=up
        
        if t-d<0 :
            y[i,t+1]=Ta[i]*y[i,t]
        elif t+1>len(t_spam)-1:
            break
        else:
            y[i,t+1]=Ta[i]*y[i,t]+(1-Ta[i])*b*u[i,t-d]
    
         
P=np.diag(np.random.randint(8,12,size=n))
P=np.eye(n)
plt.figure()
plt.title("Y in relation to time")
plt.xlabel("Time")
plt.ylabel("soil moisture")
for i in range(len(y)):
    plt.plot(y.T[:, i], label = f"Y{i+1}")
plt.legend()

fig, (ax1) = plt.subplots(ncols=1, figsize=(16, 2))
ax1.matshow(u, aspect='auto')
ax1.set_title("U in relation to time for each system")
ax1.set_ylabel('system')
ax1.set_xlabel('time')

plt.figure()
plt.plot(np.sum(P@u,axis=0))
plt.title("U en fonction de t")


for i in np.arange(0,4000):
    x=np.sum(P@u,axis=0)>1
    if np.sum(x)==0:
        break
    idx = x.argmax() // x.itemsize
    idx = idx if x[idx] else -1
    con=idx
    b=u==1
    idex=np.arange(0,n)[b[:,idx-1]]
    idex0=np.arange(0,n)[b[:,0]]
    idex=np.setdiff1d(idex,idex0)
    if not np.any(idex):
        break
    c=b[idex,:]
    c[:,:idx]=1
    nc=c.shape[0]
    id_evol=np.zeros(nc)
    for i in np.arange(0,nc):
        x=c[i,:]==0
        idx = x.argmax() // x.itemsize
        idx = idx if x[idx] else -1
        id_evol[i]=idx
        
    kmin,kargmin=np.min(id_evol),np.argmin(id_evol)
    u[idex[kargmin],:]=np.roll(u[idex[kargmin],:],-1)

fig, (ax1) = plt.subplots(ncols=1, figsize=(16, 2))
ax1.matshow(u, aspect='auto')
ax1.set_title("U in relation to time for each system")
ax1.set_ylabel('system')
ax1.set_xlabel('time')

b=2.5
for i in np.arange(0,n):
    d=Td[i]
    for t in t_spam:
        if t-d<0 :
            y[i,t+1]=Ta[i]*y[i,t]
    
        elif t+1>len(t_spam)-1:
            break
        else:
            y[i, t+1]=Ta[i]*y[i, t] +(1-Ta[i])*b*u[i, t-d]

plt.figure()
plt.plot(np.sum(P@u,axis=0))
plt.title("U en fonction de t")

plt.figure()
plt.title("Y in relation to time")
plt.xlabel("Time")
plt.ylabel("soil moisture")
for i in range(len(y)):
    plt.plot(y.T[:, i], label = f"Y{i+1}")
plt.legend()

plt.show()