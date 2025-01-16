# %%
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from hmmlearn import hmm


# %%


def simulate_neuronal_activity(num_neurons, num_states, transition_matrix, firing_rates, duration_sec, win=0.001):
    """
    Simulates neuronal activity based on transition probabilities and firing rates.

    Args:
        num_neurons (int): Number of neurons in the ensemble.
        num_states (int): Number of activity states.
        transition_matrix (np.ndarray): Transition matrix (shape: num_states x num_states).
        firing_rates (np.ndarray): Firing rates for each neuron in each state (shape: num_neurons x num_states).
        duration_sec (float): Duration of the simulation in seconds.
        win (float): The time span of the window to populate with spikes (in seconds, e.g. 0.001 for 1 msec)

    Returns:
        np.ndarray: Neuronal activity matrix (shape: num_neurons x num_time_steps).
    """
    assert duration_sec / win == int(
        duration_sec / win), "The time window for FR calculation should fit N times in the duration_sec"
    time_steps = int(duration_sec / win)  # Assuming 1 ms time resolution
    # dt = 0.001  # Time step in seconds (1 ms)

    # Initialize neuronal activity matrix
    neuronal_activity = np.zeros((num_neurons, time_steps), dtype=int)
    current_state = np.zeros((time_steps + 1), dtype=int)

    # Initial state
    current_state[0] = 0  # np.random.choice(num_states)

    for t in range(time_steps):
        # Update firing rates based on the current state
        current_firing_rates = firing_rates[current_state[t]] if len(firing_rates.shape) == 1 else firing_rates[:,
                                                                                                   current_state[t]]

        # Generate spikes for each neuron
        spikes = np.random.poisson(current_firing_rates * win)
        neuronal_activity[:, t] = spikes

        # Transition to the next state based on transition probabilities
        current_state[t + 1] = np.random.choice(num_states, p=transition_matrix[current_state[t]])
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
        for j in range(i + 1 if FF else 0, n_states):
            if i != j:
                transition_matrix[i, j] = np.random.uniform(0, 1 - diagonal_values[i])

    # Normalize rows to ensure they sum to 1
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix /= row_sums[:, np.newaxis]
    return transition_matrix


def translateToCounts(activity):
    """

        Receives binary matrix of spike times and count their number in windows of win bins.

        Args:
            activity (np.ndarray): binary matrix of spike times (MxN) N bins, M=neurons.
            win: (int): The length of the window to count spikes

        Returns:
            np.ndarray: counts of spike in bins. NxM, N bins, M=neurons
        """
    counts = activity.reshape(-1, win).sum(axis=-1)
    counts = counts.reshape(num_neurons, -1).T

    return counts


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
    bins_per_trial = int(trialssamples / trialsNum)
    bins_per_state = int(bins_per_trial / num_states)
    activities = activities[:trialsNum * bins_per_state * num_states, :]

    reshaped_data = activities.T.reshape(n_neur, trialsNum, bins_per_state, num_states)
    means = np.mean(reshaped_data, axis=(1, 2))
    return means


def fitBestPoissonModel(activities, trialsNum, num_states=3, num_iter=100):
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

    #number of samples in a trial
    ln = int(activities.shape[0] / trialsNum)
    num_neurons = activities.shape[1]
    lengths = [ln] * trialsNum
    lambdas = predictLambdas(activities, trialsNum, num_states)
    # CHECK THIS
    # non_zero_cols = [np.sum(lambdas, axis=1) > 0.3][0]
    # lambdas = lambdas[non_zero_cols,:]
    # num_neurons = np.sum(non_zero_cols==True)
    # activities = activities[:,non_zero_cols]

    for itr in range(num_iter):
        model = hmm.PoissonHMM(n_components=num_states, n_iter=100, tol=1e-4, verbose=True, init_params='', params='tl', algorithm='viterbi')
        # Assume the system alwasy start in state 0
        model.startprob_ = np.pad(np.array([1.]), (0,num_states-1), 'constant', constant_values=(0))
        # Generate a random transition metrix, either All-to-All, or feed forward
        model.transmat_ = generate_transition_matrix(num_states, FF=True)
        # Add some normal noise to the initial values of the initial firing rates
        model.lambdas_ = (lambdas + np.random.normal(loc=0, scale=lambdas / 5, size=(num_neurons, num_states))).T
        model.lambdas_[model.lambdas_<0] = 0.0001
        # Fit the data
        model.fit(activities, lengths=lengths)
        models.append(model)
        # try:
        scores.append(model.score(activities, lengths=lengths))
        # except:
        #     scores.append(float('-inf'))

        # print(f'Converged: {model.monitor_.converged}'
        #       f'\tScore: {scores[-1]}')

    # get the best model
    modelret = models[np.argmax(scores)]
    # modelret.non_zero_cols = non_zero_cols
    return modelret


def generate_activity(trialsNum, num_neurons, num_states, transition_matrix, firing_rates, duration_sec, win=200, to_FR=False):
    assert duration_sec * 1000 / win == int(duration_sec * 1000 / win), \
        "The time window for FR calculation should fit N times in the duration_sec"
    activities = np.array([[0] * num_neurons])
    states_trials = list()  # a list of state sequences, one for each trial
    lengths = np.array([], dtype=int)  # number of samples for each trial per neuron
    for i in range(trialsNum):
        activity, states = simulate_neuronal_activity(num_neurons, num_states, transition_matrix, firing_rates,
                                                      duration_sec)
        # Translate spike times (binary vectors), into spike count in bins
        counts = translateToCounts(activity)
        if to_FR==True:
            counts = counts / (win/1000)
        lengths = np.append(lengths, counts.shape[0])
        activities = np.concatenate([activities, counts])
        states_trials.append(states)

    return activities[1:, :], states_trials, lengths

def classifyBestModel(models, trial):
    '''
    Receives the firing rate of differnt neruons in a trial, and a list of models
    and return the model that best match the trial's data
    Args:
        models (dictionary): a dictionary with keys of taste names and value of
            associated model objects
        trial: (np.array): a matrix containing the firing rates of neurons in bins
                        during differnt trials. Each column contains the activity of
                        different neuron. Across the column, the trails are concatenated
                        such that the first M bins are the FR of trial 1, the next M bins
                        are of trial two, and so on

        Returns:
            hmm.model: a model object
    '''
    tastes = models['taste_order']
    scores = [models[taste].score(trial[:,models[taste].non_zero_cols]) for taste in tastes]
    ind = scores.index((max(scores)))
    return models[tastes[ind]], ind

def computeTasteCoding(dic, models, n_trials=400):
    tastes = models['taste_order']
    confusion_mat = np.zeros((len(tastes), len(tastes)))
    for n in range(n_trials):
        trial, tr_taste_ind = chooseRandomTrial(dic, tastes)
        model, classified_taste_ind = classifyBestModel(models, trial)
        confusion_mat[tr_taste_ind, classified_taste_ind]+=1
    return confusion_mat



def chooseRandomTrial(dic, tastes):
    rnd_taste = np.random.choice(tastes)
    mat = dic['data'][rnd_taste]['FR']
    lengths = dic['data'][rnd_taste]['lengths']
    samps_trial = lengths[0]
    trials_n = int(len(mat[:,0])/samps_trial)
    rnd_trial = np.random.randint(0,trials_n)
    retmat = mat[rnd_trial*samps_trial:(rnd_trial+1)*samps_trial,:]
    return retmat, tastes.index(rnd_taste)

def findStateTransitions(model, trialsdata, lengths=None):
    '''
    Using the given HMM model to predict the state transition probabilities and their timing
    :param model: Trained HMM model
    :param trialsdata: (np.array) array-like, shape (n_samples, n_features)) –
            Feature matrix of individual samples.
    :param lengths: array-like of integers, shape (n_sequences, ) – Lengths of the individual sequences in X.
            The sum of these should be n_samples
    :return:
    '''
    seqs = model.predict(trialsdata, lengths)
    trialsdata = trialsdata[:, model.non_zero_cols]
    # diff = np.diff(seqs, axis=0)
    # transitions = np.where(diff == 1)
    transition_mat = matrix = [[[] for _ in range(model.n_components)] for _ in range(model.n_components)]
    for i, lngth in enumerate(lengths):
        seq = seqs[i*lngth:(i+1)*lngth]
        for j in range(len(seq) - 1):
            transition_mat[seq[j]][seq[j+1]].append(j)

    return transition_mat


def remove_slow_neurons(activity, binwidth, minFR=1):

    activity = activity/binwidth
    meanFR = np.mean(activity, axis=0)
    non_zero_cols = np.where(meanFR > minFR)
    activity = activity[:, non_zero_cols[0]]
    return activity, non_zero_cols
def find_states_number(trialsdata, lengths, n_components=(2,3,4,5,6)):

    models = []

    for n in n_components:
        models.append(fitBestPoissonModel(trialsdata, lengths, num_states=n, num_iter=100))

    aic = [model.aic(trialsdata) for model in models]
    bic = [model.bic(trialsdata) for model in models]

    return np.argmin(aic)



if __name__ == "__main__":
    # %%
    # Example usage
    '''
    In this example there are 2 neurons, recorded for 10 seconds in each of the 10 trials. The neuronal activity is 
    governed by a 3 underlying states. The transition matrix is feed forword 

    '''
    import pickle as pkl
    dp = r'C:\Users\AnanM\OneDrive - Tel-Aviv University\Documents\TAU\data\NP\ND7_post'
    dps = r'C:\Users\AnanM\OneDrive - Tel-Aviv University\Documents\TAU\data\NP\ND7_post\ND7_post_g0'
    dpks = r'C:\Users\AnanM\OneDrive - Tel-Aviv University\Documents\TAU\data\NP\ND7_post\ND7_post_g0\kilosort3'

    simulate_exp = False
    train_models = False
    calc_transitions = False
    find_best_state_num = True

    if find_best_state_num:
        num_states = (2, 3, 4, 5, 6)
        with open(dpks + r'\ND7_sp_movingWin.pkl', 'rb') as file: packed_data = pkl.load(file)
        exp_block_dic = packed_data[0]
        taste_order = list(exp_block_dic['data'].keys())
        block_models = dict.fromkeys(taste_order)
        block_models['taste_order'] = taste_order
        for taste in taste_order:
            FR_mat_pre = exp_block_dic['data'][taste]['FR']
            # FR_mat_pre += 0.01
            # FR_mat_pre *= 100
            # CHANGE THE BINWIDTH PARAM TO SOMETHING STORED IN THE PACKED DATA
            FR_mat, non_zero_cols =  remove_slow_neurons(FR_mat_pre, binwidth=0.5, minFR=1 )
            lengths = len(exp_block_dic['data'][taste]['lengths'])
            block_models[taste] = num_states[find_states_number(FR_mat, lengths, n_components=num_states)]
            print('for taste {} the best number of states is {}'.format(taste, block_models[taste]))





    if train_models:
        with open(dpks + r'\ND7_sp_movingWin.pkl', 'rb') as file: packed_data = pkl.load(file)
        packed_data = packed_data[0:9]
        coding = []
        models = []
        for i, exp_block_dic in enumerate(packed_data):
            print('starting epoch {} !!!!!!!!!!!!!!!!!!!!!!'.format(i))
            taste_order = list(exp_block_dic['data'].keys())
            block_models = dict.fromkeys(taste_order)
            block_models['taste_order'] = taste_order
            for taste in taste_order:
                FR_mat = exp_block_dic['data'][taste]['FR']
                block_models[taste] = fitBestPoissonModel(FR_mat, len(exp_block_dic['data'][taste]['lengths']),
                                                        num_states=3, num_iter=100)
            models.append(block_models)
            coding.append(computeTasteCoding(exp_block_dic, block_models))
        with open(dpks + r'\ND7_models.pkl', 'wb') as file: pkl.dump(models, file)


    if calc_transitions:
        num_states = 3
        with open(dpks + r'\ND7_sp_movingWin.pkl', 'rb') as file:
            packed_data = pkl.load(file)
        packed_data = packed_data[0:9]
        with open(dpks + r'\ND7_models.pkl', 'rb') as file:
            models = pkl.load(file)

        fst_transition = []
        snd_transition = []
        for model in models:
            taste_order = model['taste_order']
            for t in taste_order:
                fst_trans, snd_trans = findStateTransitions(model[t], packed_data[0]['data'][t]['FR'], packed_data[0]['data'][t]['lengths'])
                fst_transition.append(fst_trans)
                snd_transition.append(snd_trans)
            a= 1

            # fst_trans, snd_trans = [transitions[0][np.where(transitions[1] == i)] for i in range(num_states)]


    if simulate_exp:
        num_neurons = 2
        num_states = 3
        duration_sec = 5.0
        win = 200

        # Example transition matrix (you can customize this)
        transition_matrix1 = np.array([[0.9995, 0.0005, 0.],
                                       [0., 0.9995, 0.0005],
                                       [0., 0., 1.0]])

        transition_matrix2 = np.array([[0.99, 0.01, 0.],
                                       [0., 0.9995, 0.0005],
                                       [0., 0., 1.0]])

        # The Emition firing rate matrix, randomly populated with some values.
        firing_rates1 = np.zeros((num_neurons, num_states))
        firing_rates1[:, 0:1] = np.random.uniform(low=1, high=7, size=(num_neurons, 1))
        firing_rates1[:, 1:2] = np.random.uniform(low=25, high=30, size=(num_neurons, 1))
        firing_rates1[:, 2:3] = np.random.uniform(low=5, high=10, size=(num_neurons, 1))

        firing_rates2 = np.zeros((num_neurons, num_states))
        firing_rates2[:, 0:1] = np.random.uniform(low=5, high=17, size=(num_neurons, 1))
        firing_rates2[:, 1:2] = np.random.uniform(low=3, high=10, size=(num_neurons, 1))
        firing_rates2[:, 2:3] = np.random.uniform(low=15, high=30, size=(num_neurons, 1))

        # Simulate neuronal activity
        # %%
        trialsNum = 10  # Number of trials

        # Responses to one taste1
        activities1, states_trials1, lengths1 = generate_activity(trialsNum, num_neurons, num_states, transition_matrix1,
                                                                  firing_rates1, duration_sec, to_FR=False)
        # Responses to one taste2
        activities2, states_trials2, lengths2 = generate_activity(trialsNum, num_neurons, num_states, transition_matrix2,
                                                                  firing_rates2, duration_sec, to_FR=False)

        rec_all, _ = activities1.shape

        # %%
        '''
            i=0
            fig, axes = plt.subplots(2,1)
            axes[0].plot(states_trials[i])
            axes[1].plot(activities[0:50])
            plt.show(block=False)
            # Save to a numpy file
            with open('activity_data.pkl', 'wb') as f:
                pickle.dump((activities, states_trials, num_neurons, num_states, transition_matrix, firing_rates, duration_sec), f)
    
            print("Neuronal activity saved to 'neuronal_activity.npy'.")
        '''
        # %%
        trlen = lengths1[0]
        Hit = Mis = 0;
        fig, axes = plt.subplots(trialsNum, 1)
        for i in range(trialsNum):

            inds_trial = range(i * trlen, (i + 1) * trlen)
            inds_rest = np.setdiff1d(range(rec_all), inds_trial)

            testTrial1 = activities1[inds_trial, :]
            testTrial2 = activities2[inds_trial, :]

            activities1_ = activities1[inds_rest, :]
            activities2_ = activities2[inds_rest, :]
            model1 = fitBestPoissonModel(activities1_, trialsNum - 1, num_states=3)
            model2 = fitBestPoissonModel(activities2_, trialsNum - 1, num_states=3)

            logprob1_1 = model1.score(testTrial1)
            logprob2_1 = model2.score(testTrial1)

            states_p1 = model1.predict(testTrial1)
            states_p2 = model2.predict(testTrial1)
            X1 = np.array([[x] * win for x in states_p1]).flatten().reshape(-1, int(duration_sec * 1000))[0]
            X2 = np.array([[x] * win for x in states_p2]).flatten().reshape(-1, int(duration_sec * 1000))[0]
            axes[i].plot(states_trials1[i], 'k')
            axes[i].plot(X1, 'b')
            axes[i].plot(X2, 'r')

            if (logprob1_1 > logprob2_1):
                Hit += 1

            else:
                Mis += 1

            logprob1_2 = model1.score(testTrial2)
            logprob2_2 = model2.score(testTrial2)
            if (logprob2_2 > logprob1_2):
                Hit += 1
            else:
                Mis += 1

            models = {'taste_order': ['t1','t2'], 't1': model1, 't2': model2}
            mod, ind = classifyBestModel(models, testTrial1)

        plt.show()
        print("Hits = {}, Misses = {}".format(Hit / trialsNum / 2, Mis / trialsNum / 2))
