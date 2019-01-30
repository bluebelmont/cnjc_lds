"""Utility for common operations on data
"""

__author__ = 'Blue Sheffer'

import os

import numpy as np
import scipy.io as scio
import sparse
import pandas as pd

class PerturbationData(object):
    def __init__(self, data_dir="../data/", filepaths=[], exclude_early=True):
        self.exclude_early = True
        self.data_dir = data_dir
        self.filepaths = filepaths
        self.sessions = []
        if len(self.filepaths) == 0:
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith('.mat'):
                        self.filepaths.append(os.path.join(root, f))
                        self.sessions = self.load_sessions(self.filepaths)
                    break
                if len(self.sessions) > 0: break
        self.delay_align_spike_times()
        self.filter_trials()
        self.add_spikes()
        self.filter_neurons()
        # convert each session object to a pandas dataframe
        # df_sessions = [pd.DataFrame(session) for session in self.sessions]
        # self.sessions = df_sessions


    def filter_trials(self):
        for session in self.sessions:
            valid_trials = np.ones(session['neurons'].shape[0],)
            # remove early behavior trials
            valid_trials = np.logical_and(valid_trials, np.logical_not(np.squeeze(session['behavior_early_report'])))
            # remove long delays
            delay_times = session['task_cue_time'][:,0] - (session['task_pole_time'][:,1] + .1)
            delay_time_std = np.std(delay_times)
            too_long_trials = delay_times - delay_times.mean() > 2 * delay_time_std
            valid_trials = np.logical_and(valid_trials, np.logical_not(too_long_trials))
            print("Filtering {} trials".format(np.sum(np.logical_not(valid_trials))))
            for key, val in session.items():
                if isinstance(val, np.ndarray):
                    session[key] = val[valid_trials]

    def load_sessions(self, filepaths):
        """Load each experimental session into an array.

        
        Args:
            filepaths (list of str): Paths to data files
            data_dir (str): Name of directory with data
            
        Returns:
            sessions (list): List where each element is the raw data from an experimental session.
        """
        sessions = []
        for f in filepaths:
            data = scio.loadmat(f)
            del data['behavior_touch_times']  # this is just a 0x0 vector, getting rid of it for now
            data['file'] = f
            sessions.append(data)
        return sessions


    def add_spikes(self):
        """Creates an attribute for each dataset with sparse spike.

        Args:
            bin_length (int, optional): Length, in milliseconds, for each bin of the spike raster.
                Defaults to 1. 
        """
        bin_length = 1
        for session in self.sessions:
            # for now, the number of timesteps between sessions can be different
            max_delay_length = np.max(session['task_cue_time'][:,0] - (session['task_pole_time'][:,1] + .1))
            num_timesteps = int(max_delay_length/(bin_length/float(1000)))
            spikes_times = session['neurons']
            neuron_coords = np.array([])
            timestep_coords = np.array([])
            trial_coords = np.array([])
            for i_trial in range(spikes_times.shape[0]):
                for i_neuron in range(spikes_times.shape[1]):
                    trial_spike_times = spikes_times[i_trial, i_neuron, 0]
                    if len(trial_spike_times) > 0:
                        trial_raster_indices = np.round((trial_spike_times / max_delay_length ) * num_timesteps)
                        timestep_raster_indices = np.ones_like(trial_raster_indices) * i_trial
                        neuron_raster_indices = np.ones_like(trial_raster_indices) * i_neuron
                        neuron_coords = np.append(neuron_coords, neuron_raster_indices)
                        trial_coords = np.append(trial_coords, trial_raster_indices)
                        timestep_coords = np.append(timestep_coords, timestep_raster_indices)
            coords = np.array((neuron_coords, timestep_coords, trial_coords),dtype=np.int32)
            data = 1
            session['spikes'] = sparse.COO(coords, data)

    def delay_align_spike_times(self):
        """ Delay aligns spikes and formats the neural data nicely

        Only keeps data from each trial's delay period

        """
        for session in self.sessions:
            neurons = session['neuron_single_units']
            num_neurons = neurons.shape[0]
            neuron_info = session['neuron_unit_info']
            num_trials = neurons[0][0].shape[0]
            all_trials = []
            for trial in range(num_trials):
                this_trial = []
                delay_start = session['task_pole_time'][trial,1]+ .1
                delay_end = session['task_cue_time'][trial,0]
                for neuron in range(num_neurons):
                    spike_times = np.squeeze(neurons[neuron][0][trial][0])
                    info = neuron_info[neuron]
                    delay_spike_times = spike_times[np.logical_and(spike_times > delay_start, spike_times < delay_end)]
                    delay_aligned = delay_spike_times - delay_start
                    this_trial.append([delay_aligned, info])
                all_trials.append(this_trial)
            session['neurons'] = np.squeeze(np.array(all_trials))
            del session['neuron_single_units']
            del session['neuron_unit_info']

    def filter_neurons(self):
        for session in self.sessions:
            spikes = session['spikes']
            total_spikes = np.sum(spikes, axis=(1, 2)).todense()
            spike_cutoff_percentage = 20
            spike_cutoff = np.sort(total_spikes)[(total_spikes.shape[0] * spike_cutoff_percentage) // 100]

            spike_filter = total_spikes > spike_cutoff
            spikes = spikes[spike_filter]
            session['spike_filter'] = spike_filter
            session['spikes'] = spikes


def naive_smooth(matrix, bin_size, step_size):
    if matrix.ndim == 2:
        matrix = np.expand_dims(matrix, 1)
    num_channels, num_trials, num_timesteps = matrix.shape
    num_smooth_timesteps = (num_timesteps - bin_size)//step_size
    smooth_matrix = np.zeros((num_channels, num_trials, num_smooth_timesteps))

    for t in range(num_smooth_timesteps):
        start, end = t*step_size, t*step_size + bin_size
        smooth_matrix[:, :, t] = matrix[:,:,start:end].mean(axis=2)
    return np.squeeze(smooth_matrix)


def find_max_time(data_dir, filenames):
    """Find the max spike time given a list of filenames
        
    Args:
        filenames (list of str): List of filenames.
        data_dir (str): Name of directory with data.

    """
    for filename in filenames:    
        data = scio.loadmat(os.path.join(data_dir, filenames))
        spikes = data['neuron_single_units']
        for i, neuron in enumerate(spikes):
            trials = neuron[0]
            all_times = []
            for trial in trials:
                if len(trial[0]):
                    all_times.append(trial[0][0][-1])
            if len(all_times):
                max_times.append(np.max(all_times))