from tbp.monty.frameworks.models.graph_matching import MontyForGraphMatching
from py4j.java_gateway import JavaGateway, GatewayParameters

class ALHTMBase(MontyForGraphMatching):
    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        super().__init__(*args, **kwargs)

        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(address='172.17.96.1', port=25333))
        self.alhtm = self.gateway.entry_point
        self.alhtm.report("Initializing Python Hooks")

    def step(self, observations, *args, **kwargs):
        super.step(observations, *args, **kwargs)
        self.alhtm.report(str(observations))
