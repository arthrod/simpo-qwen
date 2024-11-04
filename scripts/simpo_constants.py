from dataclasses import dataclass

@dataclass
class QwenConstants:
    BOS_TOKEN_ID: int = 151643
    EOS_TOKEN_IDS: list = [151645, 151643]
    PAD_TOKEN_ID: int = 151643
    MODEL_MAX_LENGTH: int = 131072
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.8
    TOP_K: int = 20
    REPETITION_PENALTY: float = 1.05
