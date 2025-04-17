import json

from py4j.java_gateway import JavaGateway, GatewayParameters

from tbp.monty.frameworks.actions.actions import (
    Action,
#    ActionJSONDecoder,
#    ActionJSONEncoder,
#    LookDown,
#    LookUp,
#    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
#    SetAgentPose,
#    SetSensorRotation,
#    TurnLeft,
#    TurnRight,
#    VectorXYZ,
)
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching, GraphLM, GraphMemory
from tbp.monty.frameworks.models.motor_policies import SurfacePolicyCurvatureInformed

gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
alhtm = gateway.entry_point

class ALHTMBase(MontyForGraphMatching):
    """ AL HTM Monty class - used for overall processing of observations? """
    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

        alhtm.report("Initializing Python ALHTMBase")

    def step(self, observations, *args, **kwargs):
        alhtm.report(str(observations))
        super(MontyForGraphMatching, self).step(observations, *args, **kwargs)

    @property
    def is_motor_only_step(self):
        return False

class ALHTMMotorSystem(SurfacePolicyCurvatureInformed):
    """ AL HTM Motor System class - Interfaces with HTM system running externally to determine movement based on observations. """
    def __init__(self, *args, **kwargs):
        """Initialize and reset motor system."""
        super().__init__(*args, **kwargs)

        alhtm.report("Initializing Python ALHTMMotorSystem")

        self.action = None
        self.is_predefined = False  # required by base class
        self.state = {}  # this must be set externally

    def dynamic_call(self) -> Action:
        # TODO: wtf fix or remove if not needed:
        # self.alhtm.report(json.dumps(self._prepare_input()))
        features = self.processed_observations.non_morphological_features
        if "mean_depth" in features:
            json_action_str = alhtm.getNextAction()
            self.action = self.build_action_from_java(json.loads(json_action_str))
            return self.action
        return None

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
    
        else:
            raise ValueError(f"Unknown action type from Java: {action_type}")


class NoOpLearningModule(GraphLM):
    """A no-op Learning Module that satisfies GraphLM interface without learning."""
    def __init__(self, initialize_base_modules=False):
        super().__init__(initialize_base_modules=initialize_base_modules)

        # Provide dummy components
        self.graph_memory = GraphMemory()
        self.graph_memory.get_initial_hypotheses = lambda: ([], [])
        self.graph_memory.load_state_dict = lambda _: None

        self.GSG = None
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