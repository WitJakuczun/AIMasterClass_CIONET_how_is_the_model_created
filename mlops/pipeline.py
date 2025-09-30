
from abc import ABC, abstractmethod
from typing import Any, List

class PipelineContext:
    """A simple object to pass data between pipeline steps."""
    def __init__(self):
        self.data = {}

    def set(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str) -> Any:
        return self.data.get(key)

class PipelineStep(ABC):
    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        pass

class Pipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, initial_context: PipelineContext) -> PipelineContext:
        context = initial_context
        for step in self.steps:
            context = step.run(context)
        return context
