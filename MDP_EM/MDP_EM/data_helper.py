#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'


import csv
import datetime
import numpy as np
import os
import pickle
import dill
import time

# Paths used for loading and saving variables.
mdp_obj_path = os.path.abspath('pickled_mdps')
episode_path = os.path.abspath('pickled_episodes')
infered_mdps_path = os.path.abspath('pickled_inference')
infered_statistics_path = os.path.abspath('pickled_inference_set_stats')

def getOutFile(name_prefix='EM_MDP', dir_path=mdp_obj_path):
    """
    @brief Create a time_stamped filename.

    @param name_prefix The leading string in the filename.
    @param dir_path The directory in which to save the file. If it doesn't exist it will be created.

    @return The full filepath to the timestamped filename.
    """
    current_datetime = datetime.datetime.now()
    formatted_time = current_datetime.strftime('_%y%m%d_%H%M')
    # Filepath for mdp objects.
    full_file_path = os.path.join(dir_path, name_prefix + formatted_time)
    if not os.path.exists(os.path.dirname(full_file_path)):
        try:
            os.makedirs(os.path.dirname(full_file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return full_file_path

def pickleMDP(variables_to_save=[], name_prefix=""):
    """
    @brief Save a list of robot MDP variables to a file in @c mdp_obj_path.

    @param variables_to_save A list of variables to pickle
    @param name_prefix A string prepended to the filename saved in @c mdp_obj_path
    """
    mdp_file = getOutFile(name_prefix)
    with open(mdp_file, 'w+') as _file:
        print "Pickling {} to {}".format(name_prefix, mdp_file)
        pickle.dump(variables_to_save, _file)
    return mdp_file

def pickleEpisodes(variables_to_save, name_prefix, num_episodes, steps_per_episode):
    """
    @brief Save variables related to history/demonstration set.

    @param variables_to_save A list of variables to pickle
    @param name_prefix A string prepended to the filename saved in @c mdp_obj_path
    @param num_episodes The number of episodes in run histories (used for filename).
    @param steps_per_episode Number of time-steps per episode (used for filename).
    """
    history_file = getOutFile(os.path.basename(name_prefix)
                              + ('_HIST_{}eps{}steps'.format(num_episodes, steps_per_episode)), episode_path)
    with open(history_file, 'w+') as _file:
        print "Pickling Episode histories to {}.".format(history_file)
        pickle.dump(variables_to_save, _file)
    return history_file

def picklePolicyInferenceMDP(variables_to_save, name_prefix):
    """
    @brief Save a list of variables from policy inference to a file in @c infered_mdps_path.

    @param variables_to_save A list of variables to pickle
    @param name_prefix A string prepended to the filename saved in @c infered_mdps_path.
    """
    infered_mdp_file = getOutFile(os.path.basename(name_prefix) + '_Policy', infered_mdps_path)
    with open(infered_mdp_file, 'w+') as _file:
        print "Pickling Infered Policy to {}.".format(infered_mdp_file)
        pickle.dump(variables_to_save, _file)
    return infered_mdp_file

def pickleInferenceStatistics(variables_to_save, name_prefix):
    """
    @brief Save a list of variables for analyzing policy inference statistics to a file in @c infered_statistics_path.

    @param variables_to_save A list of variables to pickle
    @param name_prefix A string prepended to the filename saved in @c infered_statistics_path.
    """
    infered_stats_file = getOutFile(os.path.basename(name_prefix) + '_Inference_Stats', infered_statistics_path)
    with open(infered_stats_file, 'w+') as _file:
        print "Pickling Inference Statistics to {}.".format(infered_stats_file)
        pickle.dump(variables_to_save, _file)
    return infered_stats_file

def loadPickledMDP(load_from_file):
    """
    @brief loads pickled MDP variables from a file.

    @Note this method will automatically look in @c mdp_obj_path for @c load_from_file

    @param load_from_file The filename containing variables.

    @return list_to_unpack A tuple containing all variables pickled into the file. The user is responsible for
            determining the length of the tuple and unpacking the values correctly.
    """
    mdp_file = os.path.join(mdp_obj_path, load_from_file)
    print "Loading file {}.".format(mdp_file)
    with open(mdp_file) as _file:
        list_to_unpack = pickle.load(_file)
        if not isinstance(list_to_unpack, list):
            list_to_unpack = [list_to_unpack]
    list_to_unpack.append(mdp_file)
    return list_to_unpack

def loadPickledEpisodes(load_from_file):
    """
    @brief loads pickled episode variables from a file.

    @Note this method will automatically look in @c episode_path for @c load_from_file

    @param load_from_file The filename containing variables.

    @return list_to_unpack A tuple containing all variables pickled into the file. The user is responsible for
            determining the length of the tuple and unpacking the values correctly.
    """
    pass
    history_file = os.path.join(episode_path, load_from_file)
    print "Loading history data file {}.".format(history_file)
    with open(history_file) as _file:
        list_to_unpack = pickle.load(_file)
        if not isinstance(list_to_unpack, list):
            list_to_unpack = [list_to_unpack]
    list_to_unpack.append(history_file)
    return list_to_unpack

def loadPickledPolicyInferenceMDP(load_from_file):
    """
    @brief loads pickled InfernceMDP variables from a file.

    @Note this method will automatically look in @c infered_mdps_path for @c load_from_file

    @param load_from_file The filename containing variables.

    @return list_to_unpack A tuple containing all variables pickled into the file. The user is responsible for
            determining the length of the tuple and unpacking the values correctly.
    """
    infered_mdp_file = os.path.join(infered_mdps_path, load_from_file)
    print "Loading infered policy data file {}.".format(infered_mdp_file)
    with open(infered_mdp_file) as _file:
        list_to_unpack = pickle.load(_file)
        if not isinstance(list_to_unpack, list):
            list_to_unpack = [list_to_unpack]
    list_to_unpack.append(infered_mdp_file)
    return list_to_unpack

def loadPickledInferenceStatistics(load_from_file):
    """
    @brief loads pickled Inference statistical variables from a file.

    @Note this method will automatically look in @c infered_statistics_path for @c load_from_file

    @param load_from_file The filename containing variables.

    @return list_to_unpack A tuple containing all variables pickled into the file. The user is responsible for
            determining the length of the tuple and unpacking the values correctly.
    """
    infered_stats_file = os.path.join(infered_statistics_path, load_from_file)
    print "Loading inference statistics file {}.".format(infered_stats_file)
    with open(infered_stats_file) as _file:
        list_to_unpack = pickle.load(_file)
        if not isinstance(list_to_unpack, list):
            list_to_unpack = [list_to_unpack]
    list_to_unpack.append(infered_stats_file)
    return list_to_unpack

def writePolicyToCSV(policy_dict, policy_keys_to_print=None, file_name='policy'):
    """
    @brief Write the policy dictionary to a CSV file.

    @param policy_dict A dictionary with with key-value pairs @c (state_tuple: action_dict), where @c action_dict is a
           dictionary of key-values paris @c (action_string: probability).
    @param policy_keys_to_print A list of keys to print, if None, then all keys are saved.
    @param file_name The file name to save CSV to.
    """
    if policy_keys_to_print is not None:
        csv_dict = {key: policy_dict[key] for key in policy_keys_to_print}
    else:
        csv_dict = policy_dict
    with open(file_name + '.csv', 'w+') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in csv_dict.items():
            for subkey, sub_value in value.items():
                writer.writerow([key,subkey,sub_value])

def getSmallestNumpyUnsignedIntType(max_value):
    """
    @brief Identify the smallest unsigned integer data type that can represent a range of values.

    @param max_value The maximum integer value that needs to be represented.
    """
    if max_value < np.iinfo(np.uint8).max:
        smallest_dtype = np.uint8
    elif max_value < np.iinfo(np.uint16).max:
        smallest_dtype = np.uint16
    elif max_value < np.iinfo(np.uint32).max:
        smallest_dtype = np.uint32
    elif max_value < np.iinfo(np.uint64).max:
        smallest_dtype = np.uint64
    else:
        raise ValueError('An input of max_value larger than {} is not supported.'.format(np.iinfo(np.uint64).max))
    return smallest_dtype


def printHistoryAnalysis(run_histories, states, labels, empty, goal_state):

    num_episodes = run_histories.shape[0]
    steps_per_episode = run_histories.shape[1]

    # Determine which states are goals or obstacles.
    normal_states = {state: True if label==empty else False for state, label in labels.items()}
    unique, starting_counts = np.unique(run_histories[:,0], return_counts=True)
    num_trials_from_state = {state:0 for state in states}
    num_trials_from_state.update(dict(zip(unique, starting_counts)))
    num_rewards_from_state = {state:0 for state in states}
    for run_idx in range(num_episodes):
        starting_state = run_histories[run_idx][0]
        final_state = run_histories[run_idx][-1]
        if final_state==goal_state:
            num_rewards_from_state[starting_state] += 1
    print("In this demonstration 'history' there are  {} episodes, each with {} moves.".format(num_episodes,
          steps_per_episode))
    for state in range(len(states)):
        reward_likelihood = float(num_rewards_from_state[state]) / float(num_trials_from_state[state]) if \
            num_trials_from_state[state] > 0 else np.nan
        print("State {}: Num starts = {}, Num Rewards = {}, likelihood = {}.".format(state,
                                                                                     num_trials_from_state[state],
                                                                                     num_rewards_from_state[state],
                                                                                     reward_likelihood))

