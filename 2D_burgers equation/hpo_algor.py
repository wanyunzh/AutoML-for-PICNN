
search_space={
  "UNARY_OPS": {"_type": "choice", "_value": ['absolute','square']},
  "WEIGHT_INIT": {"_type": "choice", "_value": ['one','zero']},
  "WEIGHT_OPS": {"_type": "choice", "_value": ['max','normalize','one','adaptive']},
}
from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python loss_func_search.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'minimize'
}

experiment.config.trial_concurrency = 1
experiment.run(8075)


