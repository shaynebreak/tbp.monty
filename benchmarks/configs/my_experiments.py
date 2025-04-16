# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from dataclasses import asdict

from benchmarks.configs.names import MyExperiments, YcbExperiments
from tbp.monty.frameworks.config_utils.config_args import ALHTMMontyConfig
import copy

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)
al_htm_obj_recog_experiment = copy.deepcopy(YcbExperiments["base_10simobj_surf_agent"])
al_htm_obj_recog_experiment.update(
    monty_config=ALHTMMontyConfig
)

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    al_htm_obj_recog_experiment=al_htm_obj_recog_experiment
)
CONFIGS = asdict(experiments)
