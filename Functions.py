import numpy as np 

def generate_X_spikes(T,dt,rx,N):

    return np.random.binomial(1,rx*dt,(N,int(T/dt)))/dt




def LIF_simulation(spikes_in,w=0.9,V_th=1,dt=0.1 * 10**(-3) ,tau=20 * 10**(-3),reset=True,plot=False):
    V=np.zeros(spikes_in.shape[1])
    Data_Plot=np.array([0,0])
    
    spikes_out = np.zeros(spikes_in.shape[1])
    Total_Input_spike = np.sum(spikes_in,0)
    Lenght = len(V)
    for k in range (1,Lenght):
        V[k]=V[k-1]+ dt * (-V[k-1]/tau + w * Total_Input_spike[k-1])

        if V[k]>V_th:

            if plot:
                Data_Plot=np.vstack([Data_Plot,[k*dt,V[k]]])
            if reset:
                V[k]=0

            spikes_out[k]=1/dt
        
    if plot:
        return V,spikes_out,Data_Plot[1:,:]
    else:
        return V, spikes_out

def mean_var(X_activity,K_array,dt,Transient=0.1):
    mean_practical=[]
    var_practical=[]
    N=X_activity.shape[0]
    for K in K_array:
        K_val = np.random.choice(N,K,replace=False)
        potential, _ = LIF_simulation(X_activity[K_val,:],w=1/K,reset=False)
        mean_practical.append(np.mean(potential[int(Transient/dt):]))
        var_practical.append(np.var(potential[int(Transient/dt):]))
    
    return mean_practical,var_practical

def get_Fano(Spikes,dt,Window=0.1):
    count = np.zeros(len(Spikes)-int(Window/dt))
    for j in range(len(Spikes)-int(Window/dt)):
        count[j]= np.sum(Spikes[j:j+int(Window/dt)])*dt
    
    Fano=np.var(count)/np.mean(count)
    return Fano

def generate_Connection(N=1000,K=100):
    J_ee, J_ie, J_ei, J_ii, J_ex, J_ix = 1, 1, -2, -1.8, 1, 0.8
    div = 1/np.sqrt(K)


    sequence = np.concatenate((np.ones(K),np.zeros(N-K)))
    if N-K-1 <= 0:
        diagonal = np.ones(K-1)
    else:
        diagonal = np.concatenate((np.ones(K),np.zeros(N-K-1)))
        
    E = np.zeros((N,N,3),dtype='float')
    I = np.zeros((N,N,3),dtype='float')
    

    for i in range(N):
        #E connection
        E[i,:,0] = np.insert(np.random.permutation(diagonal),i,0)*(J_ee*div)
        E[i,:,1] = np.random.permutation(sequence)*(J_ei*div)
        E[i,:,2] = np.random.permutation(sequence)*(J_ex*div)

        #I connection
        I[i,:,0] = np.random.permutation(sequence)*(J_ie*div)
        I[i,:,1] = np.insert(np.random.permutation(diagonal),i,0)*(J_ii*div)
        I[i,:,2] = np.random.permutation(sequence)*(J_ix*div)
    
    return E,I

def Network_simulation(X_activity,E_connect,I_connect,dt=0.1*10**(-3),T=2,tau = 20 * 10**(-3),V_th=1,N=1000):
    t = np.arange(0,T+dt,dt)

    E_potential = np.zeros((N,len(t)))
    E_spikes = np.zeros(E_potential.shape,dtype='int32')

    I_potential =np.zeros(E_potential.shape)
    I_spikes =np.zeros(E_potential.shape,dtype='int32')

    for i  in range(1,len(t)):
        # Membrane potential in E
        E_input = np.matmul(E_connect[:,:,0],E_spikes[:,i-1]) + np.matmul(E_connect[:,:,1],I_spikes[:,i-1]) + np.matmul(E_connect[:,:,2],X_activity[:,i-1])

        E_potential[:,i] = E_potential[:,i-1] + dt * (-E_potential[:,i-1]/tau + E_input)       
        E_spikes[:,i]=E_potential[:,i] > V_th
        E_potential[:,i] = E_potential[:,i] * (1-E_spikes[:,i])
        E_spikes[:,i]=E_spikes[:,i]/dt

        #Membrane potential in I
        I_input = np.matmul(I_connect[:,:,0],E_spikes[:,i-1]) + np.matmul(I_connect[:,:,1],I_spikes[:,i-1]) + np.matmul(I_connect[:,:,2],X_activity[:,i-1])

        I_potential[:,i] = I_potential[:,i-1] + dt * (-I_potential[:,i-1]/tau + I_input)
        I_spikes[:,i]=I_potential[:,i] > V_th
        
        I_potential[:,i] = I_potential[:,i] * (1-I_spikes[:,i])
        I_spikes[:,i]=I_spikes[:,i]/dt

        if i%2000==0:
            print(i)
    
        
    return E_potential,E_spikes,I_potential,I_spikes
