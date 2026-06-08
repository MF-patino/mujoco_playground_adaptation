import controller.plots as plots
from controller.plots import PLOT_DATA_DIR
from worldModel.common import MODELS_ROOT
import pickle, os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

plotFiles = os.listdir(PLOT_DATA_DIR)
with open(os.path.join(PLOT_DATA_DIR, "adaptScratch.pkl"), 'rb') as f:
    plotData = AttrDict(pickle.load(f))
environments = ["BlockedKnee", "RoughTerrain", "SlipperyTerrain"]

plots.plotTransferLearningAggregated(
    env_names=environments, 
    limit_evals=75,    # Adjust based on how long you let them run
    num_trials=20
)
plots.policyEmbeddings3D(plotData)
for gp_state in plotData.gp_states:
    plots.plotGPSearchHorizontal(plotData, gp_state)
for env_change in plotData.env_changes:
    plots.plotGaitPattern(plotData, env_change)
plots.statisticDriftHistory(plotData)

for pol_name in os.listdir(MODELS_ROOT):
    if "AdaptedFrom" in pol_name:
        env_name, base_pol_name = pol_name.split("_AdaptedFrom_")
        plots.transferLearningComparison(env_name, pol_name, base_pol_name)

plots.wmErrorHistory(plotData)