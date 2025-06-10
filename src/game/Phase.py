from enum import Enum


class Phase(Enum):
    STARTING_REGION = 1
    PLACE_ARMIES = 2
    ATTACK_TRANSFER = 3
    END_MOVE = 4
