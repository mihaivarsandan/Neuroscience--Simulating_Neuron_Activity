import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import scipy.stats as stats

def raster_plot(data,dt,title='Spike PLot',xlabel='Time (sec)',ylabel='Neuron Nr.'):
    plt.figure()
    y,x=np.nonzero(data)
    if data.shape[0]>20:
        s=1
        color='blue'
    else:
        s=100
        color='blue'
    plt.scatter(x*dt,y,c=color,linewidths=2,marker="|",s=s)
    plt.xlim(0,data.shape[1]*dt)
    plt.ylim(0,data.shape[0])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.show()

def LIF_plot(input_spikes,dt,V_th,membrane, output_spikes, above_threshold):
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    time_vec = (np.arange(0, len(membrane))*dt)
 

    ax1.plot(time_vec, membrane, '-', linewidth=1, label='Potential') 
    ax1.plot(time_vec, V_th*np.ones(len(time_vec)), '--', linewidth=1, label='Threshold') 
    ax1.plot(above_threshold[:,0], above_threshold[:,1], 'D', label='Neuron will fire')
    ax1.set_xlim(0, 2)
    #ax1.set_title("Membrane potential of a LIF neuron") 
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Potential") 
    ax1.legend(prop={'size': 7},loc=1)

    ax2.plot(time_vec,input_spikes, linewidth=1)
    ax2.plot(time_vec,output_spikes+1/dt, linewidth=1)
    ax2.plot(time_vec,np.zeros(len(time_vec)), 'w')
    ax2.plot(time_vec,np.ones(len(time_vec))+1/dt, 'w')
    ax2.set_xlim(0, 2)
    #ax2.set_title("Input and output spike trains") 
    ax2.set_xlabel("Time (s)")
    y_positions = [5000, 15000] 
    y_labels = ["Input", "Output"] 
    plt.yticks(y_positions, y_labels)
    plt.subplots_adjust(hspace=0.5) 
    plt.show()

def plot_membrane_Potential(Potential,V_th,dt):
    mean_val =np.mean(Potential[int(0.1/dt):])
    fig, ax1 = plt.subplots(1)
    time_vec = np.arange(0, len(Potential))*dt
    ax1.plot(time_vec, Potential, linewidth=1, label='Potential')
    ax1.plot(time_vec, V_th*np.ones(len(time_vec)), '--', linewidth=1, label='Threshold')
    ax1.plot(time_vec, mean_val*np.ones(len(time_vec)), '--',c='r', linewidth=1, label='Mean') 
    ax1.set_xlim(0, 2)
    if np.max(Potential)>V_th:
        ax1.set_ylim(0,np.max(Potential)*1.2)
    else:
        ax1.set_ylim(0,V_th*1.2)
    #ax1.set_title("Membrane potential of a LIF neuron")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Membrane potential")
    ax1.legend(loc=1)
    plt.show()

def plot_mean_var(K_th,K_pr,K_pr_ex,Mean_th,Mean_pr,Mean_pr_ex,Var_th,Var_pr,Var_pr_ex):
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(K_th, Mean_th, linewidth=2, label="Theoretical mean") 
    ax1.plot(K_pr_ex,Mean_pr_ex,'x', linewidth=2, label="Practical mean")
    ax1.plot(K_pr, Mean_pr,'x',c='r',linewidth=2, label="Practical mean")
    ax1.set_xlabel("K")
    ax1.set_ylabel("E(V[k])") 
    ax1.legend()

    ax2.loglog(K_th, Var_th, linewidth=2, label="Theoretical variance") 
    ax2.loglog(K_pr_ex, Var_pr_ex,'x', linewidth=2, label="Practical variance")
    ax2.loglog(K_pr, Var_pr,'x',c='r', linewidth=2, label="Practical variance")
    #ax2.plot(K_th, Var_th, linewidth=2, label="Theoretical variance") 
    #ax2.plot(K_pr, Var_pr, linewidth=2, label="Practical variance")
    ax2.set_xlabel("K")
    ax2.set_ylabel("Var(V[k])") 
    ax2.legend()

    plt.subplots_adjust(hspace=0.5) 
    plt.show()

def plot_normal_distribution(Potential,mean,std):
    fig, ax1 = plt.subplots(1)
    min_V = np.amin(Potential)
    max_V = np.amax(Potential)
    val = np.linspace(min_V,max_V,100)
    Normal_distribution = stats.norm.pdf(val, mean, std)
    print(Normal_distribution)
    ax1.hist(Potential, 50, density=True, alpha=0.7,label='Practical Distribution')
    ax1.plot(val,Normal_distribution,linewidth=1,label='Theoretical Distribution')
    ax1.set_xlabel("Membrane Potential")
    ax1.set_ylabel("Normalised Frequnecy of Membrane Potential values")
    ax1.legend()
    plt.show()
    




        


   