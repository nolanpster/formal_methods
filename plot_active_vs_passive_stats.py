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

# Select files with data to plot, aggregate_files are assumed to have all data in a dictionary.
active_inference_file = 'two_stage_active_stats_10_trials10_batches_5_trajs_20_stepsPerTraj_Inference_Stats_180413_1621'
passive_inference_file = \
'two_stage_passive_stats_10_trials10_batches_5_trajs_20_stepsPerTraj_Inference_Stats_180413_1543'
# Single agent interactive learning?
multi_agent = True
active_inference_file = 'two_stage_active_stats_50_trials30_batches_1_trajs_10_stepsPerTraj_Inference_Stats_180424_0225'
passive_inference_file = \
'two_stage_passive_stats_50_trials30_batches_1_trajs_10_stepsPerTraj_Inference_Stats_180424_0222'
aggregate_file = None


active_data_dict, _, = DataHelper.loadPickledInferenceStatistics(active_inference_file)
passive_data_dict, _,  = DataHelper.loadPickledInferenceStatistics(passive_inference_file)

PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_L1_norms'],
                                 val_array_2=passive_data_dict['passive_inference_L1_norms'], plot_quantiles=True,
                                 transparency=0.2)
if multi_agent:
    PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_count_of_trajs_reacing_goal'],
                                     val_array_2=passive_data_dict['passive_inference_count_of_trajs_reacing_goal'],
                                     title='Agent 1 Ends at Goal', ylabel='Count of Trajectories',
                                     plot_quantiles=True)
plt.show()
