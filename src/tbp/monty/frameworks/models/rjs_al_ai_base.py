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
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.models.motor_policies import SurfacePolicyCurvatureInformed
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule

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

class NoOpLearningModule(LearningModule):
    """A no-operation LearningModule that does nothing but satisfies interface requirements."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    ###
    # Methods that interact with the experiment
    ###
    def reset(self):
        """Do things like reset buffers or possible_matches before training."""
        pass

    def pre_episode(self, primary_target):
        """Do things like reset buffers or possible_matches before training."""
        pass

    def post_episode(self):
        """Do things like update object models with stored data after an episode."""
        pass

    def set_experiment_mode(self, mode):
        """Set the experiment mode.

        Update state variables based on which method (train or evaluate) is being called
        at the experiment level.
        """
        pass

    ###
    # Methods that define the algorithm
    ###
    def matching_step(self):
        """Matching / inference step called inside of monty._step_learning_modules."""
        pass

    def exploratory_step(self):
        """Model building step called inside of monty._step_learning_modules."""
        pass

    def receive_votes(self, votes):
        """Process voting data sent out from other learning modules."""
        pass

    def send_out_vote(self):
        """This method defines what data are sent to other learning modules."""
        pass

    def propose_goal_state(self):
        """Return the goal-state proposed by this LM's GSG."""
        pass

    def get_output(self):
        """Return learning module output (same format as input)."""
        pass

    ###
    # Saving, loading
    ###

    def state_dict(self):
        """Return a serializable dict with everything needed to save/load this LM."""
        pass

    def load_state_dict(self, state_dict):
        """Take a state dict as an argument and set state for this LM."""
        pass
