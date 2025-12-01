from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class AlgorithmParameter:
    name: str
    type: str
    default: Any
    label: str
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None
    widget: Optional[str] = None

    def to_dict(self):
        d = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label": self.label,
            "description": self.description
        }
        if self.min is not None: d["min"] = self.min
        if self.max is not None: d["max"] = self.max
        if self.step is not None: d["step"] = self.step
        if self.options is not None: d["options"] = self.options
        if self.widget is not None: d["widget"] = self.widget
        return d

@dataclass
class Algorithm:
    id: str
    name: str
    category: str # ID of the category, e.g. "load_data"
    prompt: str
    template: str
    parameters: List[AlgorithmParameter] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def to_prompt_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt
        }

@dataclass
class AlgorithmCategory:
    id: str
    label: str
    algorithms: List[Algorithm] = field(default_factory=list)

    def add_algorithm(self, algo: Algorithm):
        self.algorithms.append(algo)

    def to_prompt_dict(self):
        return {
            "label": self.label,
            "algorithms": [algo.to_prompt_dict() for algo in self.algorithms]
        }
