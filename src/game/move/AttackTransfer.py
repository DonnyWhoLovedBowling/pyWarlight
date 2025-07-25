from src.game.move.FigthResult import FightResult
from src.game.Region import Region
from dataclasses import dataclass, field


@dataclass
class AttackTransfer:
    from_region: Region
    to_region: Region
    armies: int
    result: FightResult = field(default_factory=FightResult)

    def __eq__(self, other):
        return (
            self.to_region == other.to_region
            and self.from_region == other.from_region
            and self.armies == other.armies
        )

    def set_armies(self, armies):
        self.armies = armies

    def get_from_region(self):
        return self.from_region

    def get_to_region(self):
        return self.to_region

    def get_armies(self):
        return self.armies

    def __str__(self):
        return (
            f"attack/transfer with {self.armies} armies from {self.from_region} to {self.to_region}",
        )

    def is_attack(self) -> bool:
        if self.result is None:
            ret = False
        else:
            ret = self.result.winner is not None
        return ret

