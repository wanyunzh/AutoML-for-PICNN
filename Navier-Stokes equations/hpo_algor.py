
search_space={
  # "constraint": {"_type": "choice", "_value": [0,1,2]},
  "UNARY_OPS": {"_type": "choice", "_value": [0,1,2]},
  "WEIGHT_INIT": {"_type": "choice", "_value": [0,1]},
  "WEIGHT_OPS": {"_type": "choice", "_value": [0,1,2,3]},
  
   "gradient": {"_type": "choice", "_value": [0,1]},
  "kernel": {"_type": "choice", "_value": [2,3,4,5]}
}
from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python loss_func_search.py'
# experiment.config.trial_command = 'python case3_unet.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'minimize'
}

# experiment.config.tuner.name = 'SMAC'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'minimize'
# }
# experiment.config.tuner.name = 'GridSearch'
# experiment.config.tuner.name = 'Anneal'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'minimize'
# }
# from nni.algorithms.hpo.medianstop_assessor import MedianstopAssessor
experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args = {'optimize_mode': 'minimize'
                                         }
experiment.config.trial_concurrency = 1
experiment.run(8075)


