import json
import os

from py4j.java_gateway import JavaGateway, GatewayParameters

from tbp.monty.frameworks.actions.actions import (
    Action,
#    ActionJSONDecoder,
#    ActionJSONEncoder,
#    LookDown,
#    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
#    SetSensorRotation,
#    TurnLeft,
#    TurnRight,
#    VectorXYZ,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching, GraphLM, GraphMemory
from tbp.monty.frameworks.models.motor_policies import SurfacePolicyCurvatureInformed
from tbp.monty.frameworks.models.goal_state_generation import GraphGoalStateGenerator
import numpy as np
import quaternion  # ensure this is imported

gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
# alhtm = gateway.entry_point.getAlHtm("demo")
alhtm = None
alhtm_observation_data = dict()
agent_state = dict();
SHARED_DIR = "/mnt/c/shared-data"

class ALHTMBase(MontyForGraphMatching):
    """ AL HTM Monty class - used for overall processing of observations? """
    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

        alhtm.report("Initializing Python ALHTMBase")
        alhtm.reset(42) # TODO: hook up to see from experiment somehow or another...

    def pre_episode(self, primary_target, semantic_id_to_label=None):
        super().pre_episode(primary_target, semantic_id_to_label)
        alhtm.onNewEpisode()

    def step(self, observations, *args, **kwargs):
        self.report_observation(observations)

        super(MontyForGraphMatching, self).step(observations, *args, **kwargs)

    def report_observation(self, observations):
        """ extracts and sends to HTM the requested observation(s) from the full list of observations """

        # update the agent start with position and rotation for the agent and all sensors...
        agent_state["current_position"] = self.motor_system.state[self.motor_system.agent_id]["position"]
        agent_state["current_rotation"] = self.motor_system.state[self.motor_system.agent_id]["rotation"]

        # Add all sensors' positions and rotations dynamically
        sensor_states = self.motor_system.state[self.motor_system.agent_id].get("sensors", {})
        sensor_info = {}
        
        for sensor_id, sensor_data in sensor_states.items():
            sensor_info[f"{sensor_id}_position"] = sensor_data.get("position", None)
            sensor_info[f"{sensor_id}_rotation"] = sensor_data.get("rotation", None)

        # Merge into agent_state
        agent_state.update(sensor_info)

        if(self.step_type_count == 0):
            alhtm.report(str(agent_state))

        # pull requested sensor and data from observations...
        sensor_and_type = alhtm.getObservationRequest()
        requested_observation = observations["agent_id_0"][sensor_and_type[0]][sensor_and_type[1]].tolist()

        # get or create java safe array...
        rows = len(requested_observation)
        cols = len(requested_observation[0]) if rows > 0 else 0
        self.save_raw_memmap(sensor_and_type[0], sensor_and_type[1], rows, cols, requested_observation)

        # send off to AL HTM...
        alhtm.setObservation(sensor_and_type[0], sensor_and_type[1], rows, cols)

        if(self.is_done):
            alhtm.report(str(agent_state))

    def save_raw_memmap(self, sensor_id, sensor_type, rows, cols, observation_array):
        # Ensure float64 format (double)
        dtype = np.float64
        flat_array = np.array(observation_array, dtype=dtype).flatten()

        # lookup or cache the memory mapped file...
        key = (sensor_id, sensor_type)
        if key in alhtm_observation_data:
            fp = alhtm_observation_data[key]

        else:
            # Create and cache the array
            filename = f"{sensor_id}_{sensor_type}_{rows}x{cols}.raw"
            filepath = os.path.join(SHARED_DIR, filename)

            # If the file exists, delete it to ensure shape/dtype match
            if os.path.exists(filepath):
                os.remove(filepath)

            # Write memory-mapped double data
            fp = np.memmap(filepath, dtype=dtype, mode='w+', shape=flat_array.shape)
            alhtm_observation_data[key] = fp

        fp[:] = flat_array[:]
        fp.flush()

    @property
    def is_done(self):
        return alhtm.isDone()

    @property
    def is_motor_only_step(self):
        return False

    def check_if_any_lms_updated(self):
        return True # we always update our learning module because it's HTM...

class ALHTMMotorSystem(SurfacePolicyCurvatureInformed):
    """ AL HTM Motor System class - Interfaces with HTM system running externally to determine movement based on observations. """
    def __init__(self, *args, **kwargs):
        """Initialize and reset motor system."""
        super().__init__(*args, **kwargs)

        global alhtm

        alhtm = gateway.entry_point.getAlHtm(self.args.htm_config)
        alhtm.report("Initializing Python ALHTMMotorSystem")

        self.action = None
        self.is_predefined = False  # required by base class
        self.state = {}  # this must be set externally

    def dynamic_call(self) -> Action:
        # TODO: wtf fix or remove if not needed:
        # self.alhtm.report(json.dumps(self._prepare_input()))
        # features = self.processed_observations.non_morphological_features
        # if "mean_depth" in features:
        json_action_str = alhtm.getNextAction()
        self.action = self.build_action_from_java(json.loads(json_action_str))
        return self.action
        # return MoveForward(self.agent_id, 0.0)

    def predefined_call(self):
        raise NotImplementedError("This policy does not support predefined actions.")

    def post_action(self, action: Action) -> None:
        # Store or log the action
        self.action = action

    def set_experiment_mode(self, mode):
        # No-op for now
        pass

    def last_action(self) -> Action:
        return self.action

    @property
    def is_motor_only_step(self):
        agent_state = self.state.get(self.agent_id, {})
        return agent_state.get("motor_only_step", False)

    def _prepare_input(self):
        # Return minimal input structure expected by Java
        return {
            "agent_id": self.agent_id,
            "state": self.state.get(self.agent_id, {})
        }

    def build_action_from_java(self, action_json: dict) -> Action:
        """Build a full Action object from JSON sent by Java."""
        action_type = action_json["action"]
        agent_id = action_json["agent_id"]

        if action_type == "orient_vertical":
            rotation_degrees = action_json["rotation_degrees"]
            down_distance, forward_distance = self.vertical_distances(rotation_degrees)
            return OrientVertical(
                agent_id=agent_id,
                rotation_degrees=rotation_degrees,
                down_distance=down_distance,
                forward_distance=forward_distance,
            )

        elif action_type == "orient_horizontal":
            rotation_degrees = action_json["rotation_degrees"]
            left_distance, forward_distance = self.horizontal_distances(rotation_degrees)
            return OrientHorizontal(
                agent_id=agent_id,
                rotation_degrees=rotation_degrees,
                left_distance=left_distance,
                forward_distance=forward_distance,
            )

        elif action_type == "move_tangentially":
            distance = action_json["distance"]
            direction = action_json["direction"]
            return MoveTangentially(
                agent_id=agent_id,
                distance=distance,
                direction=direction,
            )

        elif action_type == "move_forward":
            distance = action_json["distance"]
            return MoveForward(
                agent_id=agent_id,
                distance=distance
            )

        elif action_type == "set_agent_pose":
            # Rotation delta is expected as [w, x, y, z]
            rotation_delta_list = action_json["rotation_delta"]

            current_position = self.state[agent_id]["position"]
            current_rotation = self.state[agent_id]["rotation"]
            r = quaternion.as_rotation_matrix(current_rotation);
            right = r[:, 0]
            up = r[:, 1]
            q_pitch = quaternion.from_rotation_vector(rotation_delta_list[2]*right)
            q_yaw = quaternion.from_rotation_vector(rotation_delta_list[1]*up)

            # Apply delta rotation (delta + current)
            new_rotation = q_yaw * q_pitch * current_rotation

            return SetAgentPose(
                agent_id=agent_id,
                location=current_position,
                rotation_quat=new_rotation.normalized()
            )

        else:
            raise ValueError(f"Unknown action type from Java: {action_type}")


class ALHTMLearningModule(GraphLM):
    """A no-op Learning Module that satisfies GraphLM interface without learning."""
    def __init__(self, initialize_base_modules=False):
        super().__init__(initialize_base_modules=initialize_base_modules)

        # Provide dummy components
        self.graph_memory = GraphMemory()
        self.graph_memory.get_initial_hypotheses = lambda: ([], [])
        self.graph_memory.load_state_dict = lambda _: None
        self.gsg = GraphGoalStateGenerator(self, gsg_args=None)
        self.gsg.reset()

        self.matching_buffer = None
        self.gsg_buffer = None
        self.input_feature_modules = []
        self.output_feature_modules = []

        self.learning_module_id = "NoopLM"
        self.mode = "eval"
        self.has_detailed_logger = False
        self.primary_target = None
        self.detected_object = None
        self.detected_pose = [None for _ in range(7)]
        self.terminal_state = None

    def matching_step(self, observations):
        if self.buffer:
            self.buffer.append_input_states(observations)
            self.buffer.update_stats({"noop": True}, append=self.has_detailed_logger)
            self.buffer.stepwise_targets_list.append("no_label")

    def exploratory_step(self, observations):
        if self.buffer:
            self.buffer.append_input_states(observations)

    def post_episode(self):
        pass

    def send_out_vote(self):
        return set()

    def receive_votes(self, vote_data):
        pass

    def get_possible_matches(self):
        return []

    def get_unique_pose_if_available(self, object_id):
        return None

    def set_detected_object(self, terminal_state):
        self.terminal_state = terminal_state
        self.detected_object = None
