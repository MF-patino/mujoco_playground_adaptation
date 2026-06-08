import controller.plots as plots
from controller.plots import PLOT_DATA_DIR
import pickle, os

PLOT_FILE = "adaptSequence.pkl"

plots.plotTransferLearningAggregated(
    env_names=["BlockedKnee", "RoughTerrain", "SlipperyTerrain"], 
    limit_evals=75,
    num_trials=20
)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

with open(os.path.join(PLOT_DATA_DIR, PLOT_FILE), 'rb') as f:
    plotData = AttrDict(pickle.load(f))

# Spheric embedding plot
plots.policyEmbeddings3D(plotData)

# Plotting each GP search
for gp_state in plotData.gp_states:
    plots.plotGPSearchHorizontal(plotData, gp_state)

# Plotting each gait pattern around environment changes
for env_change in plotData.env_changes:
    plots.plotGaitPattern(plotData, env_change)

# Plots all drift detections and KS statistic/p-values around it
plots.statisticDriftHistory(plotData)

# Plots WM errors 
# Only useful for demonstrating WMs don't alwyas recognize
# their native environment
#plots.wmErrorHistory(plotData)