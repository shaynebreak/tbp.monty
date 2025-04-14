from py4j.java_gateway import JavaGateway, GatewayParameters
from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import BasePolicy
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

class ALHTMMotorSystem(BasePolicy):
    def __init__(
        self,
        rng,
        action_sampler_args: Dict,
        action_sampler_class: Type[ActionSampler],
        agent_id: str,
        switch_frequency,
        file_name=None,
        file_names_per_episode=None,
    ):
        """Initialize and reset motor system."""
        super().__init__(
            rng,
            action_sampler_args,
            action_sampler_class,
            agent_id,
            switch_frequency,
            file_name,
            file_names_per_episode,
        )

        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
        self.alhtm = self.gateway.entry_point
        self.alhtm.report("Initializing Python Hooks")

        self.action = None
        self.is_predefined = False  # required by base class
        self.state = {}  # this must be set externally

    def dynamic_call(self) -> Action:
        self.alhtm.report(self._prepare_input())
        json_action = self.alhtm.getNextAction()
        self.action = self._convert_to_action(json_action)
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

    def _convert_to_action(self, json_action):
        return json.loads(json_action, cls=ActionJSONDecoder)
