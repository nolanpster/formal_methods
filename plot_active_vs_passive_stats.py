import MDP_EM.MDP_EM.data_helper as DataHelper
import MDP_EM.MDP_EM.plot_helper as PlotHelper

from copy import deepcopy
import numpy as np
from pprint import pprint
import warnings

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import cm # For color cycler
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D

# Set warning filter before pandas import so pandas recognizes it.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import pdb

########################################################################################################################
# Numpy Print Options
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

########################################################################################################################
# Select files with data to plot, aggregate_files are assumed to have all data in a dictionary.
active_inference_file = 'two_stage_active_stats_10_trials10_batches_5_trajs_20_stepsPerTraj_Inference_Stats_180413_1621'
passive_inference_file = \
'two_stage_passive_stats_10_trials10_batches_5_trajs_20_stepsPerTraj_Inference_Stats_180413_1543'
# Single agent interactive learning?
multi_agent = False
active_inference_file = \
'single_agent_active_stats_50_trials40_batches_1_trajs_2_stepsPerTraj_Inference_Stats_180429_1958'
passive_inference_file = \
'single_agent_passive_stats_50_trials40_batches_1_trajs_2_stepsPerTraj_Inference_Stats_180429_1957'
aggregate_file = None
true_optimal_policies_to_load = 'true_optimal_policies_em_15H_100N_Inference_Stats_180423_2008'



active_data_dict, _, = DataHelper.loadPickledInferenceStatistics(active_inference_file)
passive_data_dict, _,  = DataHelper.loadPickledInferenceStatistics(passive_inference_file)

PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_L1_norms'],
    val_array_2=passive_data_dict['passive_inference_L1_norms'], plot_quantiles=True, transparency=0.2,
    ylabel=r'Fractional $||\pi_2, \tilde{\pi}_2(\tilde{\mathbf{\theta}})||_{\infty}$ w.r.t max error',
    title='Single Agent Inference')
if multi_agent:
    PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_count_of_trajs_reacing_goal'],
                                     val_array_2=passive_data_dict['passive_inference_count_of_trajs_reacing_goal'],
                                     title='Agent 1 Ends at Goal', ylabel='Count of Trajectories',
                                     plot_quantiles=True)

    true_optimal_VI_policy, true_optimal_EM_policy, _, = \
        DataHelper.loadPickledInferenceStatistics(true_optimal_policies_to_load)
    EM_policy_error = np.sum(np.absolute(np.subtract(true_optimal_VI_policy, true_optimal_EM_policy)))/2/625
    pdb.set_trace()

    PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_robot_policy_err']/2/(625),
                                     val_array_2=passive_data_dict['passive_inference_robot_policy_err']/2/(625),
                                     title='', ylabel='L-inf robot', plot_quantiles=True,
                                     threshold_line=EM_policy_error,
                                     threshold_label=r'EM with true $\pi_2, Z=100, H=15$')
plt.show()
