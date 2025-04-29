from dataclasses import dataclass

from src.game.Region import Region


@dataclass
class PlaceArmies:
    region: Region
    armies: int

    def __eq__(self, other):
        return self.region == other.region and self.armies == other.armies

    def __str__(self):
        return f"place {self.armies} on  {self.region.name}"

