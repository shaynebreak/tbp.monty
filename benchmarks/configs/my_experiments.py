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
from tbp.monty.frameworks.models.motor_policy_configurators import make_curv_surface_policy_config
import copy

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)
al_integration_test_experiment = copy.deepcopy(CONFIGS["base_10simobj_surf_agent"])
al_integration_test_experiment.update(
    monty_config=ALHTMMontyConfig(),
    eval_dataloader_class=ED.EnvironmentDataLoaderPerObject
)

al_htm_center_view_experiment = copy.deepcopy(al_integration_test_experiment)
al_htm_center_view_experiment["monty_config"].motor_system_config.motor_system_args = make_curv_surface_policy_config(
    desired_object_distance=0.025,
    alpha=0.1,
    pc_alpha=0.5,
    max_pc_bias_steps=32,
    min_general_steps=8,
    min_heading_steps=12,
    use_goal_state_driven_actions=True,
    htm_config="center_view"  # override to center_view but I have to do all this other shit also somehow...
)

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    al_integration_test_experiment=al_integration_test_experiment,
    al_htm_center_view_experiment=al_htm_center_view_experiment
)
CONFIGS = asdict(experiments)
