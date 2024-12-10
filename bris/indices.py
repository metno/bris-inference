from functools import cached_property

from checkpoint import Checkpoint


class Indices:
    def __init__(self, full, forcing, diagnostic, prognostic):
        self.full = full
        self.forcing = forcing
        self.diagnostic = diagnostic
        self.prognostic = prognostic
