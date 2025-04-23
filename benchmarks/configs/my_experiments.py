# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

from benchmarks.configs.names import MyExperiments
from benchmarks.configs.ycb_experiments import CONFIGS
from tbp.monty.frameworks.config_utils.config_args import ALHTMMontyConfig
from tbp.monty.frameworks.environments import embodied_data as ED
import copy

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)
al_integration_test_experiment = copy.deepcopy(CONFIGS["base_10simobj_surf_agent"])
al_integration_test_experiment.update(
    monty_config=ALHTMMontyConfig(),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject
)

al_htm_center_view_experiment = copy.deepcopy(al_integration_test_experiment)
setattr(
    al_htm_center_view_experiment["monty_config"].motor_system_config.motor_system_args,
    "htm_config",
    "center_view"
)

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    al_integration_test_experiment=al_integration_test_experiment,
    al_htm_center_view_experiment=al_htm_center_view_experiment
)
CONFIGS = asdict(experiments)
