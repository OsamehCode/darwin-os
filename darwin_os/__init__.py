"""DARWIN-OS: a self-evolving 2D control simulator."""

from .actions import action_to_control
from .dna import DecisionTreeDNA, random_dna, mutate, crossover, ACTION_VOCAB
from .environment import DT_SECONDS, Environment, Obstacle, Vortex
from .digital_twin import DigitalTwin, TwinConfig, TwinResult
from .verifier import SafetyVerifier, SafetyRule, RuleViolation, SafetyReport, Verdict
from .evolution import EvolutionEngine, EvolutionResult, GenerationStats, EvolutionConfig
from .controller import Controller, HotSwapEvent
from .crisis_detector import CrisisDetector, CrisisAssessment

__all__ = [
    "action_to_control",
    "DecisionTreeDNA", "random_dna", "mutate", "crossover", "ACTION_VOCAB",
    "DT_SECONDS", "Environment", "Obstacle", "Vortex",
    "DigitalTwin", "TwinConfig", "TwinResult",
    "SafetyVerifier", "SafetyRule", "RuleViolation", "SafetyReport", "Verdict",
    "EvolutionEngine", "EvolutionResult", "GenerationStats", "EvolutionConfig",
    "Controller", "HotSwapEvent",
    "CrisisDetector", "CrisisAssessment",
]

__version__ = "0.1.0"
