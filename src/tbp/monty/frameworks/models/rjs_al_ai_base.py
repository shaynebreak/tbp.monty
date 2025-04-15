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
import pydevd

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
        features = self.processed_observations.non_morphological_features
        if "mean_depth" in features:
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
