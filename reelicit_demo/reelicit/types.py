from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Feature:
    name: str
    description: str


@dataclass
class EvaluatedPrompt:
    prompt: str
    score: float
    embedding: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskSpec:
    name: str
    size: int
    context: str
    format_instruction: str
    output_format: str
    input_template: str
    answer_regex: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None


@dataclass
class RunConfig:
    q: int = 5
    T: int = 6
    K: int = 5
    M: int = 10
    tau: float = 0.1
    b: int = 10
    nmax: int = 12
    pmax: int = 20
    optimizer_temperature: float = 0.7
    target_temperature: float = 0.0
    bo_num_restarts: int = 20
    bo_raw_samples: int = 512
    strict_paper: bool = False
    include_format_in_context: bool = False
    no_refinement: bool = False
    no_bo: bool = False
    static_features: bool = False
    independent_extraction: bool = False

    @property
    def N(self) -> int:
        return self.q * self.T


@dataclass
class RunResult:
    best_prompt: str
    best_score: float
    history: List[EvaluatedPrompt]
