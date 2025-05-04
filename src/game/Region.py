from dataclasses import dataclass
from typing import Self
from svgelements import Path

from src.game.Continent import Continent


@dataclass
class Region:
    path: Path
    name: str
    id: int
    continent: Continent
    neighbours: list[Self] = None
    label_position: tuple[int,int] = 0,0

    def __post_init__(self):
        self.neighbours = list()
        
    def get_id(self) -> int:
        return self.id

    def get_continent(self) -> Continent:
        return self.continent

    def get_neighbours(self) -> list[Self]:
        return self.neighbours

    def get_name(self) -> str:
        return self.name

    def get_label_position(self) -> tuple[int,int]:
        if self.label_position is None:
            raise Exception(f"Region {self.get_name()} position is None")
        return self.label_position

    def add_neighbour(self, neighbour: Self):
        self.neighbours.append(neighbour)

    def set_label_position(self, s):
        self.label_position = s

#     public Point labelPosition; // in global coordinates
#
#     private List < Region > neighbours = new ArrayList < Region > ();
#
#     def Region(self, Path svgElement, String name, int id, Continent continent) {
#     this.svgElement = svgElement;
#     this.name = name;
#     this.id = id;
#     this.continent = continent;
#     svgElement = []
#
#
#     def getId(self) -> int
#         return id;
#
#
# public
# Continent
# getContinent()
# {
# return continent;
# }
#
# public
# String
# getName()
# {
# return name;
# }
#
# public
# void
# setLabelPosition(Point
# p) {
# labelPosition = p;
# }
#
# public
# Point
# getLabelPosition()
# {
# if (labelPosition == null)
#     throw
#     new
#     Error("region '" + name + "' has no label position");
#
# return labelPosition;
# }
#
# public
# void
# addNeighbor(Region
# r) {
# if (!neighbours.contains(r))
# neighbours.add(r);
# }
#
# public
# List < Region > getNeighbors()
# {
# return neighbours;
# }
# }
