from Functions import *
from Utility import *

""" INPUT PARAMETERS """

N=1000 # numper of neurons in each population
K=100 #number of neurons  connected to the LIF neuron
r_X= 10 #firing rate of the poisson neurons in pop. X
delta_t = 0.1 * 10**(-3) #integration time step
X=8 #size of the population 
T=2 #timeline
n_bins = T/delta_t #number of bins
w = 0.9 # synaptic weight
V_th=1 #voltage threshold
tau= 20 * 10**(-3)
EXERCISE=4.1

""" END """

if EXERCISE==1:
    """EXERCISE 1 SPIKE GENERATION"""
    N=10
    Nr_of_runs = 30
    Average_vector =np.zeros(Nr_of_runs)

    for i in range(Nr_of_runs):
        X_activity=generate_X_spikes(T,delta_t,r_X,N)
        Average_vector[i]=np.mean(np.sum(X_activity,1)*delta_t/T)
    raster_plot(X_activity,delta_t,title='X population activity plot')
    print(np.mean(Average_vector),np.std(Average_vector))

    """END"""

elif EXERCISE==2:

    """EXERCISE 2 Single LIF neuron output from a single neuron"""

    np.random.seed(4)
    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    voltage_accum,output_spikes, thresh_data = LIF_simulation(X_activity[:1,:],w=w,V_th=V_th,dt=delta_t,tau=tau,reset=True,plot=True)
    LIF_plot(X_activity[0,:],delta_t,V_th,voltage_accum,output_spikes,thresh_data)

    """END"""

elif EXERCISE==3.1:
    """EXERCISE 3a Single LIF neuron output from multiple neurons"""
    w=1
    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    K_random =np.random.choice(N,K,replace=False)
    Potential_membrane,Output_Spikes = LIF_simulation(X_activity[K_random,:],w=w/K,reset=False)
    print(np.nonzero(Potential_membrane))
    plot_membrane_Potential(Potential_membrane,V_th=V_th,dt=delta_t)

    """END"""
elif EXERCISE==3.2:
    """EXERCISE 3c Single LIF neuron output from multiple neurons"""
    w=1
    T=10
    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    K_array = np.array([1,10,100,1000])
    K_array_extra=np.array([5,25,50,75,250,500,750])
    
    mean_practice, var_practice =mean_var(X_activity=X_activity,K_array=K_array,dt=delta_t)
    mean_practice_extra, var_practice_extra =mean_var(X_activity=X_activity,K_array=K_array_extra,dt=delta_t)
    
    K_val=np.arange(0,N+1,10)
    K_val[0]=1
    mean_theoretical = tau * w * r_X * np.ones(len(K_val))
    var_theoretical = tau/2 * w**2 * r_X / K_val
    #var_theoretical =(tau**2)/(2*tau-delta_t) * w**2 * r_X /K_val

    plot_mean_var(K_val,K_array,K_array_extra,mean_theoretical,mean_practice,mean_practice_extra,var_theoretical,var_practice,var_practice_extra)

    """END"""

elif EXERCISE==3.3:
    """EXERCISE 3d Single LIF neuron output from multiple neurons"""
   
    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    K_random =np.random.choice(N,K,replace=False)
    w=V_th/(tau*r_X)
    mean = tau * w * r_X
    var = tau/2 * w**2 * r_X / K
    Potential_membrane,Output_Spikes = LIF_simulation(X_activity[K_random,:],w=w/K,reset=False)
    
    plot_membrane_Potential(Potential_membrane,V_th=V_th,dt=delta_t)
    plot_normal_distribution(Potential_membrane[int(0.1/delta_t):],mean,np.sqrt(var))

    """END"""
elif EXERCISE==3.4:
    """EXERCISE 3e Single LIF neuron output from multiple neurons"""

    Nr_of_runs=8
    #w=4.29
    w_array=np.arange(1,5,0.1) 
    T=10
    rates=np.zeros((len(w_array),Nr_of_runs))
    #Fanos=[]
    #rates=[]
    for i,w in enumerate(w_array):
        for num in range(Nr_of_runs):
            X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
            K_val =np.random.choice(N,K,replace=False)
            _,output_spikes = LIF_simulation(spikes_in=X_activity[K_val,:],w=w/K)
            rates[i,num]=np.sum(output_spikes)*delta_t/T
            #rates.append(np.sum(output_spikes)*delta_t/T)
            #Fanos.append(get_Fano(output_spikes,dt=delta_t))
    
    rates_sum =np.mean(rates,1)
    rates_plot =np.abs(rates_sum -r_X * np.ones(rates_sum.shape))
    fig, ax1 = plt.subplots(1)
    ax1.plot(w_array,rates_plot,linewidth=1)
    ax1.set_xlabel("w")
    ax1.set_ylabel("$|r_{out}- 10|$ (Hz)")
    plt.show()
    

    #print("Weight",w)
    #print("Firing Rate",np.mean(rates),np.std(rates))
    #print("Fano Factor",np.mean(Fanos),np.std(Fanos))

    """END"""

elif EXERCISE==4:
    """EXERCISE 4 Single LIF neuron with many E and I Poisson inputs """
    Nr_of_runs=30
    w=1.63
    #w_array=np.arange(1,2,0.01)
    #rates=np.zeros((len(w_array),Nr_of_runs))
    T=10
    rates=[]
    Fanos=[]
    #for i,w in enumerate(w_array):
    for num in range(Nr_of_runs):
        X_activity= generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
        K_inh_val = np.random.choice(N,K,replace=False)
        K_exh_val = np.random.choice(N,K,replace=False)
        Combined_EXH_INH = np.concatenate((X_activity[K_exh_val,:],-1*X_activity[K_inh_val,:]))
        _,Output_Spikes=LIF_simulation(spikes_in=Combined_EXH_INH,w=w/np.sqrt(K))
        rates.append(np.sum(Output_Spikes)*delta_t/T)
        Fanos.append(get_Fano(Output_Spikes,dt=delta_t))
        #rates[i,num]=np.sum(Output_Spikes)*delta_t/T
    
    """
    rates_sum =np.mean(rates,1)
    rates_plot =np.abs(rates_sum -r_X * np.ones(rates_sum.shape))
    fig, ax1 = plt.subplots(1)
    ax1.plot(w_array,rates_plot,linewidth=1)
    ax1.set_xlabel("w")
    ax1.set_ylabel("$|r_{out}- 10|$ (Hz)")
    plt.show()
    """
    
    print("Weight",w)
    print("Firing Rate",np.mean(rates),np.std(rates))
    print("Fano Factor",np.mean(Fanos),np.std(Fanos))

    """END"""
elif EXERCISE==4.1:
    w=1.8
    X_activity= generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    K_inh_val = np.random.choice(N,K,replace=False)
    K_exh_val = np.random.choice(N,K,replace=False)
    Combined_EXH_INH = np.concatenate((X_activity[K_exh_val,:],-1*X_activity[K_inh_val,:]))
    Potential_membrane,__=LIF_simulation(spikes_in=Combined_EXH_INH,w=w/np.sqrt(K),reset=False)

    mean = 0
    var = tau * w**2 * r_X

    plot_membrane_Potential(Potential_membrane,V_th=V_th,dt=delta_t)
    plot_normal_distribution(Potential_membrane[int(0.1/delta_t):],mean,np.sqrt(var))

    


elif EXERCISE==5.1:
    """EXERCISE 5b Full Network """

    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    E_connectivity,I_connectivity = generate_Connection()
    E_potential,E_spikes,I_potential,I_spikes = Network_simulation(X_activity,E_connectivity,I_connectivity)
    rate_E= np.average(np.sum(E_spikes,1)*delta_t/T)
    rate_I= np.average(np.sum(I_spikes,1)*delta_t/T)
    print("E rate:",rate_E)
    print("I rate:",rate_I)

    """END"""

elif EXERCISE==5.2:
    """EXERCISE 5c Full Network """
    r_x_array=[5,10,15,20]
    rates=np.zeros((4,2))
    E_connectivity,I_connectivity = generate_Connection()

    for i,r_x in enumerate(r_x_array):
        print(r_x)
        X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_x,N=N)
        E_potential,E_spikes,I_potential,I_spikes = Network_simulation(X_activity,E_connectivity,I_connectivity)
        rates[i,0]= np.average(np.sum(E_spikes,1)*delta_t/T)
        rates[i,1]= np.average(np.sum(I_spikes,1)*delta_t/T)

    print(rates)

    """END"""

elif EXERCISE==5.3:
    """EXERCISE 5d Full Network """

    N=100
    X_activity=generate_X_spikes(T=T,dt=delta_t,rx=r_X,N=N)
    E_connectivity,I_connectivity = generate_Connection(N=N)
    E_potential,E_spikes,I_potential,I_spikes = Network_simulation(X_activity,E_connectivity,I_connectivity,N=N)
    rate_E= np.average(np.sum(E_spikes,1)*delta_t/T)
    rate_I= np.average(np.sum(I_spikes,1)*delta_t/T)
    print("E rate:",rate_E)
    print("I rate:",rate_I)
    raster_plot(E_spikes,delta_t,title='E population activity plot')

    """END"""







