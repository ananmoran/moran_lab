#%%
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from hmmlearn import hmm
#%%


def simulate_neuronal_activity(num_neurons, num_states, transition_matrix, firing_rates, duration_sec):
    """
    Simulates neuronal activity based on transition probabilities and firing rates.
    
    Args:
        num_neurons (int): Number of neurons in the ensemble.
        num_states (int): Number of activity states.
        transition_matrix (np.ndarray): Transition matrix (shape: num_states x num_states).
        firing_rates (np.ndarray): Firing rates for each neuron in each state (shape: num_neurons x num_states).
        duration_sec (float): Duration of the simulation in seconds.
    
    Returns:
        np.ndarray: Neuronal activity matrix (shape: num_neurons x num_time_steps).
    """
    time_steps = int(duration_sec * 1000)  # Assuming 1 ms time resolution
    dt = 0.001  # Time step in seconds (1 ms)
    
    # Initialize neuronal activity matrix
    neuronal_activity = np.zeros((num_neurons, time_steps), dtype=int)
    current_state = np.zeros((time_steps+1), dtype=int)

    # Initial state 
    current_state[0] = 0 #np.random.choice(num_states)
    
    for t in range(time_steps):
        # Update firing rates based on the current state
        current_firing_rates = firing_rates[current_state[t]] if len(firing_rates.shape)==1 else firing_rates[:, current_state[t]]
        
        # Generate spikes for each neuron
        spikes = np.random.poisson(current_firing_rates * dt)
        neuronal_activity[:, t] = spikes
        
        # Transition to the next state based on transition probabilities
        current_state[t+1] = np.random.choice(num_states, p=transition_matrix[current_state[t]])
    current_state = current_state[:-1]
    return neuronal_activity, current_state

def generate_transition_matrix(n_states, FF=True):
    """
    Generates a semi-random transition matrix with specified properties.

    Args:
        n_states (int): Number of states.
        FF: (boolean): True if Feedforward transition matrix, False for all to all

    Returns:
        np.ndarray: Transition matrix (n_states x n_states) with random values.
    """
    # Generate random diagonal values between 0.95 and 0.99999
    diagonal_values = np.random.uniform(0.95, 0.99999, size=n_states)

    # Create an empty matrix
    transition_matrix = np.zeros((n_states, n_states))

    # Fill the diagonal with the random values
    np.fill_diagonal(transition_matrix, diagonal_values)

    # Fill the off-diagonal elements with random values (excluding the diagonal)
    for i in range(n_states):
        for j in range(i+1 if FF else 0,n_states):
            if i != j:
                transition_matrix[i, j] = np.random.uniform(0, 1 - diagonal_values[i])

    # Normalize rows to ensure they sum to 1
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix /= row_sums[:, np.newaxis]

    return transition_matrix

def translateToFR(activity, win):
    """
        Receives binary matrix of spike times and count their number in windows of win bins.

        Args:
            activity (np.ndarray): binary matrix of spike times (MxN) N bins, M=neurons.
            win: (int): The length of the window to count spikes

        Returns:
            np.ndarray: counts of spike in bins. NxM, N bins, M=neurons
        """
    activityFR = activity.reshape(-1, win).sum(axis=-1) / (win / 1000)
    activityFR = activityFR.reshape(num_neurons, -1).T

    return activityFR

def predictLambdas(activities, trialsNum, num_states):
    """
        Predict the underlying poisson distribution parameter lambda for each neurons in each of the num_states states.
        Assumes a feed forward state sequence.

        Args:
            activity (np.ndarray): spike count matrix (NxM) ,N=Trials_num*bins_per_trial, M=neurons.
            trialsNum: (int): Number of trials
            num_states: (int): Number of desired states

        Returns:
            np.ndarray: Matrix of lambdas. MxN, M=neurons, N=states
        """

    trialssamples, n_neur = activities.shape
    bins_per_trial = int(trialssamples/trialsNum)
    bins_per_state = int(bins_per_trial/num_states)
    reshaped_data = activities.T.reshape(n_neur, trialsNum, bins_per_state, num_states)
    means = np.mean(reshaped_data, axis=(1,2))
    return means

def getBestPoissonModel(activities, trialsNum, num_states=3, num_iter=100):
    """
            Predict the best Poisson medel parametes given the neuronal activity data.

            Args:
                activity (np.ndarray): spike count matrix (NxM) ,N=Trials_num*bins_per_trial, M=neurons.
                trialsNum: (int): Number of trials
                num_states: (int): Number of desired states, default: 3

            Returns:
                hmm.PoissonHMM: a model object
            """

    scores = list()
    models = list()

    ln = int(activities.shape[0]/trialsNum)
    lengths = [ln] * trialsNum
    lambdas = predictLambdas(activities, trialsNum, num_states)

    for itr in range(num_iter):
        model = hmm.PoissonHMM(n_components=3, n_iter=100, verbose=False, init_params='')
        # Assume the system alwasy start in state 0
        model.startprob_ = np.array([1.0, 0., 0.])
        # Generate a random transition metrix, either All-to-All, or feed forward
        model.transmat_ = generate_transition_matrix(num_states, FF=True)
        # Add some normal noise to the initial values of the initial firing rates
        model.lambdas_ = (lambdas + np.random.normal(loc=0, scale=lambdas / 5, size=(num_neurons, num_states))).T

        # Fit the data
        model.fit(activities, lengths=lengths)
        models.append(model)
        scores.append(model.score(activities, lengths=lengths))
        print(f'Converged: {model.monitor_.converged}'
              f'\tScore: {scores[-1]}')

    # get the best model
    modelret = models[np.argmax(scores)]
    return modelret


if __name__ == "__main__":
# Example usage
    '''
    In this example there are 2 neurons, recorded for 10 seconds in each of the 10 trials. The neuronal activity is 
    governed by a 3 underlying states. The transition matrix is feed forword 
    
    '''
    # activities_test = np.array([[1]*20, [2]*20, [3]*20, [4]*20, [5]*20, [6]*20, [7]*20, [8]*20, [9]*20, [10]*20]).flatten().reshape(2, -1).T
    # predictLambdas(activities_test, 5, 4)

    num_neurons = 2
    num_states = 3
    duration_sec = 9.0

    # Example transition matrix (you can customize this)
    transition_matrix = np.array([[0.9995, 0.0005, 0.],
                                  [0.,  0.9995,  0.0005],
                                  [0.,  0.,   1.0]])

    # The Emition firing rate matrix, randomly populated with some values.
    firing_rates = np.zeros((num_neurons, num_states))


    firing_rates[:,0:1] = np.random.uniform(low=1, high=7, size=(num_neurons, 1))
    firing_rates[:,1:2] = np.random.uniform(low=25, high=30, size=(num_neurons, 1))
    firing_rates[:,2:3] = np.random.uniform(low=5, high=10, size=(num_neurons, 1))


    # Simulate neuronal activity
    #%%
    trialsNum = 10   # Number of trials
    activities = np.array([[0]*num_neurons])
    states_trials = list()  # a list of state sequences, one for each trial
    lengths = np.array([],dtype=int)  # number of samples for each trial per neuron
    win = 200 # ms of windows to count spikes
    for i in range(trialsNum):
        activity, states = simulate_neuronal_activity(num_neurons, num_states, transition_matrix, firing_rates, duration_sec)
        # Translate spike times (binary vectors), into spike count in bins
        activityFR = translateToFR(activity, win)
        lengths = np.append(lengths, activityFR.shape[0])
        activities = np.concatenate([activities, activityFR])
        states_trials.append(states)

    activities = activities[1:, :]

    #%%
    '''
    i=0
    fig, axes = plt.subplots(2,1)
    axes[0].plot(states_trials[i])
    axes[1].plot(activities[0:50])
    plt.show(block=False)
    #%%
    # Save to a numpy file
    with open('activity_data.pkl', 'wb') as f:
        pickle.dump((activities, states_trials, num_neurons, num_states, transition_matrix, firing_rates, duration_sec), f)
    
    print("Neuronal activity saved to 'neuronal_activity.npy'.")
    '''
    # %%

    # X = [[0.5, 2.4], [1.0, 4.2], [-1.0, 0.5], [0.42, -0.24], [0.24, 0.25]]
    # lengths = [len(X)]
    # model = hmm.GaussianHMM(n_components=3).fit(X, lengths)
    # states_p = model.predict(activities[6][:, None].reshape(-1,1))

    model = getBestPoissonModel(activities, trialsNum, num_states=3)
    length = int(activities.shape[0]/trialsNum)
    lengths = [length] * trialsNum
    states_p = model.predict(activities, lengths=lengths)
    X = np.array([[x]*win for x in states_p]).flatten().reshape(-1,int(duration_sec*1000))

    fig, axes = plt.subplots(trialsNum, 1)

    for tri in range(trialsNum):
        axes[tri].plot(states_trials[tri], 'b')
        axes[tri].plot(X[tri,:], 'r')
    plt.show()

    # %%
