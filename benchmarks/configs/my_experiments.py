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
from tbp.monty.frameworks.config_utils.policy_setup_utils import make_informed_policy_config
from tbp.monty.frameworks.actions.action_samplers import (
    ConstantSampler
)
import copy

# Add your experiment configurations here
# e.g.: my_experiment_config = dict(...)
al_integration_test_experiment = copy.deepcopy(CONFIGS["base_config_10distinctobj_dist_agent"])
al_integration_test_experiment_args = copy.deepcopy(al_integration_test_experiment["experiment_args"])
al_integration_test_experiment_args.update(
        do_eval=True,
        n_eval_epochs=2,
        max_eval_steps=14000,
        max_total_steps=14000
)
al_integration_test_experiment.update(
    experiment_args=al_integration_test_experiment_args,
    monty_config=ALHTMMontyConfig(),
)

al_htm_center_view_experiment = copy.deepcopy(al_integration_test_experiment)
al_htm_center_view_experiment["monty_config"].motor_system_config.motor_system_args = make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=True,
            htm_config="center_view"  # override to center_view but I have to do all this other shit also somehow...
        )
al_htm_center_view_orbit_experiment = copy.deepcopy(al_integration_test_experiment)
al_htm_center_view_orbit_experiment["monty_config"].motor_system_config.motor_system_args = make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=True,
            htm_config="center_view_orbit"  # override to center_view but I have to do all this other shit also somehow...
        )
al_htm_obj_recognition_experiment = copy.deepcopy(al_integration_test_experiment)
al_htm_obj_recognition_experiment["monty_config"].motor_system_config.motor_system_args = make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=True,
            htm_config="obj_recog"  # override to obj_recog but I have to do all this other shit also somehow...
        )
al_htm_obj_recognition_v2_experiment = copy.deepcopy(al_integration_test_experiment)
al_htm_obj_recognition_v2_experiment["monty_config"].motor_system_config.motor_system_args = make_informed_policy_config(
            action_space_type="distant_agent_no_translation",
            action_sampler_class=ConstantSampler,
            rotation_degrees=5.0,
            use_goal_state_driven_actions=True,
            htm_config="obj_recog_v2"  # override to obj_recog but I have to do all this other shit also somehow...
        )

experiments = MyExperiments(
    # For each experiment name in MyExperiments, add its corresponding
    # configuration here.
    # e.g.: my_experiment=my_experiment_config
    al_integration_test_experiment=al_integration_test_experiment,
    al_htm_center_view_experiment=al_htm_center_view_experiment,
    al_htm_center_view_orbit_experiment=al_htm_center_view_orbit_experiment,
    al_htm_obj_recognition_experiment=al_htm_obj_recognition_experiment,
    al_htm_obj_recognition_v2_experiment=al_htm_obj_recognition_v2_experiment
)
CONFIGS = asdict(experiments)
