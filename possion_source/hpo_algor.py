
search_space={
  "constraint": {"_type": "choice", "_value": [0,1,2]},
  "loss function": {"_type": "choice", "_value": [0,1,2,3]},
 "kernel": {"_type": "choice", "_value": [2,3,4,5]},
}
from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python loss_func_search.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space

# experiment.config.tuner.name = 'GP'
# experiment.config.tuner.class_args = {
#     'optimize_mode': 'minimize',
#     'utility': 'ei',
#     'kappa': 5.0,
#     'xi': 0.0,
#     'nu': 2.5,
#     'alpha': 1e-6,
#     'cold_start_num': 10,
#     'selection_num_warm_up': 100000,
#     'selection_num_starting_points': 250
# }

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args = {
    'optimize_mode': 'minimize'
}


from nni.algorithms.hpo.medianstop_assessor import MedianstopAssessor
experiment.config.assessor.name = 'Medianstop'
experiment.config.assessor.class_args = {'optimize_mode': 'minimize',
                                        'start_step': 10
                                      }
experiment.config.trial_concurrency = 1
experiment.run(8061)


