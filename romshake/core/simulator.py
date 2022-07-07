from romshake.simulators.analytic import AnalyticSimulator
from romshake.simulators.seissol_simulate import SeisSolSimulator


class Simulator():
    def __init__(self, sim_type, **kwargs):
        if sim_type == 'analytic':
            self.simulator = AnalyticSimulator(**kwargs)
        elif sim_type == 'seissol':
            self.simulator = SeisSolSimulator(**kwargs)
