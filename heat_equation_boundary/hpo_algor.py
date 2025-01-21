
search_space={
  "constraint": {"_type": "choice", "_value": [0,1,2]},
  "UNARY_OPS": {"_type": "choice", "_value": [0,1,2]},
  "WEIGHT_INIT": {"_type": "choice", "_value": [0,1]},
  "WEIGHT_OPS": {"_type": "choice", "_value": [0,1,2,3]},
   "gradient": {"_type": "choice", "_value": [0,1]},
  "kernel": {"_type": "choice", "_value": [2,3,4,5]}
}
from nni.experiment import Experiment
experiment = Experiment('local')
# experiment = Experiment('local', experiment_name='my_experiment')

experiment.config.trial_command = 'python loss_func_search.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

# Try hpo_gp.py or use this one for BO
# experiment.config.tuner.name = 'GP'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'minimize',
#     'utility': 'ei',
#     'kappa': 5.0,
#     'xi': 0.0,
#     'nu': 2.5,
#     'alpha': 1e-6,
#     'cold_start_num': 1,
#     'selection_num_warm_up': 200,
#     'selection_num_starting_points': 250
# }
## minimal config ##

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'minimize'
}
# experiment.config.tuner.name = 'SMAC'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'minimize'
# }

# from nni.algorithms.hpo.medianstop_assessor import MedianstopAssessor
# experiment.config.assessor.name = 'Medianstop'
# experiment.config.assessor.class_args = {'optimize_mode': 'minimize',
#                                         'start_step': 2
#                                       }
experiment.config.trial_concurrency = 1
experiment.run(8070)


