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
aggregate_file = None


active_data_dict, _, = DataHelper.loadPickledInferenceStatistics(active_inference_file)
passive_data_dict, _,  = DataHelper.loadPickledInferenceStatistics(passive_inference_file)

PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_L1_norms'],
                                 val_array_2=passive_data_dict['passive_inference_L1_norms'], plot_quantiles=True,
                                 plot_min_max=True)

PlotHelper.plotValueStatsVsBatch(val_array_1=active_data_dict['active_inference_fraction_of_trajs_reacing_goal'],
                                 val_array_2=passive_data_dict['passive_inference_fraction_of_trajs_reacing_goal'],
                                 title='Agent 1 Ends at Goal', ylabel='Fraction of Trajectories', plot_quantiles=True,
                                 plot_min_max=True)
plt.show()