import sys

if sys.version_info[1] < 11:
    from typing_extensions import Self
else:
    from typing import Self

from dataclasses import dataclass
# from svgelements import Path

from src.game.Continent import Continent


@dataclass
class Region:
    # path: Path
    name: str
    id: int
    continent: Continent
    neighbours: list[Self] = None
    label_position: tuple[int, int] = 0, 0

    def __post_init__(self):
        self.neighbours = list()
        self.continent.add_region(self)

    def get_id(self) -> int:
        return self.id

    def get_continent(self) -> Continent:
        return self.continent

    def get_neighbours(self) -> list[Self]:
        return self.neighbours

    def get_name(self) -> str:
        return self.name

    def get_label_position(self) -> tuple[int, int]:
        if self.label_position is None:
            raise Exception(f"Region {self.get_name()} position is None")
        return self.label_position

    def add_neighbour(self, neighbour: Self):
        self.neighbours.append(neighbour)

    def set_label_position(self, s):
        self.label_position = s

    def __repr__(self):
        return self.name
