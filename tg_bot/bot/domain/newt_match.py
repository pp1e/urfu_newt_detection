from dataclasses import dataclass


@dataclass(frozen=True)
class NewtMatch:
    class_name: str
    similarity: float
