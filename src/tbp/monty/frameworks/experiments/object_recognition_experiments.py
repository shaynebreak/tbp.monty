# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from tbp.monty.frameworks.environments.embodied_data import SaccadeOnImageDataLoader
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

from .monty_experiment import MontyExperiment
import wandb

# turn interactive plotting off -- call plt.show() to open all figures
plt.ioff()


class MontyObjectRecognitionExperiment(MontyExperiment):
    """Experiment customized for object-pose recognition with a single object.

    Adds additional logging of the target object and pose for each episode and
    specific terminal states for object recognition. It also adds code for
    handling a matching and an exploration phase during each episode when training.

    Note that this experiment assumes a particular model configuration, in order
    for the show_observations method to work: a zoomed out "view_finder"
    rgba sensor and an up-close "patch" depth sensor
    """

    def run_episode(self):
        """Episode that checks the terminal states of an object recognition episode."""
        self.pre_episode()
        last_step = self.run_episode_steps()
        self.post_episode(last_step)

    def pre_episode(self):
        """Pre-episode hook.

        Pre episode where we pass the primary target object, as well as the mapping
        between semantic ID to labels, both for logging/evaluation purposes.
        """
        # TODO, eventually it would be better to pass
        # self.dataloader.semantic_id_to_label via an "Observation" object when this is
        # eventually implemented, such that we can ensure this information is never
        # inappropriately accessed and used
        if hasattr(self.dataloader, "semantic_id_to_label"):
            self.model.pre_episode(
                self.dataloader.primary_target, self.dataloader.semantic_id_to_label
            )
        else:
            self.model.pre_episode(self.dataloader.primary_target)
        self.dataloader.pre_episode()

        self.max_steps = self.max_train_steps
        if not self.model.experiment_mode == "train":
            self.max_steps = self.max_eval_steps

        self.logger_handler.pre_episode(self.logger_args)

        if self.show_sensor_output:
            self.initialize_online_plotting()

    def post_episode(self, steps):
        if self.show_sensor_output:
            self.cleanup_online_plotting()
        super().post_episode(steps)

    def run_episode_steps(self):
        """Runs one episode of the experiment.

        At each step, observations are collected from the dataloader and either passed
        to the model or sent directly to the motor system. We also check if a terminal
        condition was reached at each step and increment step counters.

        Returns:
            The number of total steps taken in the episode.
        """
        for loader_step, observation in enumerate(self.dataloader):
            if(self.force_exit):
                break;

            if self.show_sensor_output:
                self.show_observations(observation, loader_step)

            if self.model.check_reached_max_matching_steps(self.max_steps):
                logging.info(
                    f"Terminated due to maximum matching steps : {self.max_steps}"
                )
                # Need to break here already, otherwise there are problems
                # when the object is recognized in the last step
                return loader_step

            if loader_step >= (self.max_total_steps):
                logging.info(f"Terminated due to maximum episode steps : {loader_step}")
                self.model.deal_with_time_out()
                return loader_step

            if self.model.is_motor_only_step:
                logging.debug(
                    "Performing a motor-only step, so passing info straight to motor"
                )
                # On these sensations, we just want to pass information to the motor
                # system, so bypass the main model step (i.e. updating of LMs)
                self.model.pass_features_directly_to_motor_system(observation)
            else:
                self.model.step(observation)

            if self.model.is_done:
                # Check this right after step to avoid setting time out
                # after object was already recognized.
                return loader_step
        # handle case where spiral policy calls StopIterator in motor policy
        self.model.set_is_done()

        # if we're quitting quickly exit...
        if(self.force_exit):
            wandb.finish()
            self.close()
            raise SystemExit

        return loader_step

    def initialize_online_plotting(self):
        if hasattr(self, "fig") and self.fig:
            # Figure already exists; reuse
            return

        plt.ion()  # Enable interactive mode just once
        self.fig, self.ax = plt.subplots(
            1, 2, figsize=(9, 6), gridspec_kw={"width_ratios": [1, 0.8]}
        )
        self.fig.subplots_adjust(top=1.1)

        # escape key handler...
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
        self.force_exit = False

        # Initialize placeholders for dynamic updates
        self.camera_image = self.ax[0].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
        self.depth_image = self.ax[1].imshow(np.zeros((10, 10)), cmap="viridis_r")
        self.setup_camera_ax()
        self.setup_sensor_ax()

    def cleanup_online_plotting(self):
        if self.show_sensor_output:
            # No need to close the window
            self.camera_image.set_data(np.zeros_like(self.camera_image.get_array()))
            self.depth_image.set_data(np.zeros_like(self.depth_image.get_array()))

    # Add this method in your class
    def handle_key_press(self, event):
        if event.key == 'escape':
            print("[INFO] ESC pressed — exiting.")
            self.force_exit = True
            plt.close('all')  # Close all matplotlib windows
            sys.exit(0)       # Exit the program

    def show_observations(self, observation, step):
        action_name = getattr(self.dataloader, "_action", None)
        if action_name is not None:
            action_name = action_name.__class__.__name__
        else:
            action_name = "Unknown"

        self.fig.suptitle(
            f"Observation at step {step}"
            + ("" if step == 0 else f"\n{action_name}")
        )
        self.show_view_finder(observation, step)
        self.show_patch(observation)
        plt.pause(0.00001)

    def show_view_finder(self, observation, step, sensor_id="view_finder"):
        view_finder_image = observation[self.model.motor_system.agent_id][sensor_id]["rgba"]
    
        # Optionally modify image with patch outline
        if isinstance(self.dataloader, SaccadeOnImageDataLoader):
            center_pixel_id = np.array([200, 200])
            patch_size = np.array(
                observation[self.model.motor_system.agent_id]["patch"]["depth"]
            ).shape[0]
            raw_obs = self.model.sensor_modules[0].raw_observations
            if len(raw_obs) > 0:
                center_pixel_id = np.array(raw_obs[-1]["pixel_loc"])
            view_finder_image = add_patch_outline_to_view_finder(
                view_finder_image, center_pixel_id, patch_size
            )
    
        # Initialize imshow once
        if self.camera_image is None:
            self.camera_image = self.ax[0].imshow(view_finder_image, zorder=-99)
    
            # Draw square once
            if step == 0 and not hasattr(self, "_patch_square_added"):
                image_shape = view_finder_image.shape
                square = plt.Rectangle(
                    (image_shape[1] * 4.5 // 10, image_shape[0] * 4.5 // 10),
                    image_shape[1] / 10,
                    image_shape[0] / 10,
                    fc="none",
                    ec="white",
                )
                self.ax[0].add_patch(square)
                self._patch_square_added = True
        else:
            self.camera_image.set_data(view_finder_image)
    
        # Add optional text overlay
        if hasattr(self.model.learning_modules[0].graph_memory, "current_mlh"):
            mlh = self.model.learning_modules[0].get_current_mlh()
            if mlh is not None:
                self.add_text(mlh, pos=view_finder_image.shape[0])

    def show_patch(self, observation, sensor_id="patch"):
        # patch_image = observation[self.model.motor_system.agent_id][sensor_id]["rgba"]
        #
        # if self.depth_image is None:
        #     self.depth_image = self.ax[1].imshow(patch_image)
        # else:
        #     self.depth_image.set_data(patch_image)
        patch_image = observation[self.model.motor_system.agent_id][sensor_id]["depth"]
        
        if self.depth_image is None:
            self.depth_image = self.ax[1].imshow(patch_image, cmap="viridis_r")
        else:
            self.depth_image.set_data(patch_image)
            self.depth_image.set_clim(vmin=patch_image.min(), vmax=patch_image.max())

    def add_text(self, mlh, pos):
        if self.text:
            self.text.remove()
        new_text = r"MLH: "
        mlh_id = mlh["graph_id"].split("_")
        for word in mlh_id:
            new_text += r"$\bf{" + word + "}$ "
        new_text += f"with evidence {np.round(mlh['evidence'],2)}\n\n"
        pms = self.model.learning_modules[0].get_possible_matches()
        graph_ids, evidences = self.model.learning_modules[
            0
        ].graph_memory.get_evidence_for_each_graph()

        # Highlight 2nd MLH if present
        if len(evidences) > 1:
            top_indices = np.flip(np.argsort(evidences))[0:2]
            second_id = graph_ids[top_indices[1]].split("_")
            new_text += "2nd MLH: "
            for word in second_id:
                new_text += r"$\bf{" + word + "}$ "
            new_text += f"with evidence {np.round(evidences[top_indices[1]],2)}\n\n"

        new_text += r"$\bf{Possible}$ $\bf{matches:}$"
        for gid, ev in zip(graph_ids, evidences):
            if gid in pms:
                new_text += f"\n{gid}: {np.round(ev,1)}"

        self.text = self.ax[0].text(0, pos + 30, new_text, va="top")

    def setup_camera_ax(self):
        self.ax[0].set_title("Camera image")
        self.ax[0].set_axis_off()
        self.camera_image = None
        self.text = None

    def setup_sensor_ax(self):
        self.ax[1].set_title("Sensor depth image")
        self.ax[1].set_axis_off()
        self.depth_image = None


class MontyGeneralizationExperiment(MontyObjectRecognitionExperiment):
    """Remove the tested object model from memory to see what is recognized instead."""

    def pre_episode(self):
        """Pre episode where we pass target object to the model for logging."""
        if "model.pt" not in self.model_path:
            model_path = os.path.join(self.model_path, "model.pt")
        state_dict = torch.load(model_path)
        print(f"loading models again from {model_path}")
        self.model.load_state_dict(state_dict)
        super().pre_episode()
        target_object = self.dataloader.primary_target["object"]
        print(f"removing {target_object}")
        for lm in self.model.learning_modules:
            lm.graph_memory.remove_graph_from_memory(target_object)
            print(f"graphs in memory: {lm.get_all_known_object_ids()}")
