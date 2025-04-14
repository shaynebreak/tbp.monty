from py4j.java_gateway import JavaGateway, GatewayParameters
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import SurfacePolicyCurvatureInformed
from tbp.monty.frameworks.actions.action_samplers import ActionSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONDecoder,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorRotation,
    TurnLeft,
    TurnRight,
    VectorXYZ,
)
from typing import Any, Callable, Dict, List, Mapping, Tuple, Type, Union, cast
import json
import numpy as np

class ALHTMBase(MontyForGraphMatching):
    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
        self.alhtm = self.gateway.entry_point
        self.alhtm.report("Initializing Python Hooks")

    def step(self, observations, *args, **kwargs):
        self.alhtm.report(str(observations))
        return super.step(observations, *args, **kwargs)

class ALHTMMotorSystem(SurfacePolicyCurvatureInformed):
    def __init__(self, *args, **kwargs):
        """Initialize and reset motor system."""
        super().__init__(*args, **kwargs)

        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
        self.alhtm = self.gateway.entry_point
        self.alhtm.report("Initializing Python Hooks")

        self.action = None
        self.is_predefined = False  # required by base class
        self.state = {}  # this must be set externally

    def dynamic_call(self) -> Action:
        # TODO: wtf fix or remove if not needed:
        # self.alhtm.report(json.dumps(self._prepare_input()))
        json_action_str = self.alhtm.getNextAction()
        self.action = self.build_action_from_java(json.loads(json_action_str))
        return self.action

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

    def build_action_from_java(self, action_json: dict):
        """
        Given partial JSON input from Java, return a fully constructed Action instance.
        Uses Monty's internal state to compute necessary parameters like depth.
    
        Args:
            action_json (dict): JSON with at least 'action' and 'agent_id'.
    
        Returns:
            Action: The completed Action object.
        """
    
        action_type = action_json["action"]
        agent_id = action_json["agent_id"]
    
        # Get mean depth from sensor module (assumes surface policy structure)
        features = self.processed_observations.non_morphological_features
        mean_depth = features.get("mean_depth", 1)
    
        def orient_vertical_handler():
            rotation_degrees = action_json["rotation_degrees"]
            rotation_radians = np.radians(rotation_degrees)
    
            down_distance = np.tan(rotation_radians) * mean_depth
            forward_distance = (
                mean_depth * (1 - np.cos(rotation_radians)) / np.cos(rotation_radians)
            )
    
            return OrientVertical(
                agent_id=agent_id,
                rotation_degrees=rotation_degrees,
                down_distance=down_distance,
                forward_distance=forward_distance,
            )
    
        def orient_horizontal_handler():
            rotation_degrees = action_json["rotation_degrees"]
            rotation_radians = np.radians(rotation_degrees)
    
            left_distance = np.tan(rotation_radians) * mean_depth
            forward_distance = (
                mean_depth * (1 - np.cos(rotation_radians)) / np.cos(rotation_radians)
            )
    
            return OrientHorizontal(
                agent_id=agent_id,
                rotation_degrees=rotation_degrees,
                left_distance=left_distance,
                forward_distance=forward_distance,
            )
    
        # Switch/case style dispatch
        action_dispatch = {
            "orient_vertical": orient_vertical_handler,
            "orient_horizontal": orient_horizontal_handler,
        }
    
        if action_type not in action_dispatch:
            raise ValueError(f"Unsupported or unknown action type from Java: {action_type}")
    
        return action_dispatch[action_type]()
