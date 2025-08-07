import logging

from line_profiler_pycharm import profile
import os
from src.game.Continent import Continent
from src.game.Region import Region

def do_bounding_boxes_intersect(bbox1, bbox2):
    """Checks if two bounding boxes (min_x, min_y, max_x, max_y) intersect."""
    if bbox1 is None or bbox2 is None:
        return False
    return not (
        bbox1[2] < bbox2[0]
        or bbox1[0] > bbox2[2]
        or bbox1[3] < bbox2[1]
        or bbox1[1] > bbox2[3]
    )

class World:

    def __init__(self, map_name):
        self.continents: list[Continent] = []
        self.regions: list[Region] = []
        self.torch_edge_list: list[list[int]] = []
        # self.diagram: SVG = SVG()
        map_file = f"{map_name}.txt"
        if os.path.isfile(map_file):
            continent_map = dict()
            region_map = dict()
            infile = open(map_file, "r")
            read_continents = True
            read_regions = False
            read_edges = False
            froms = []
            tos = []
            for line in infile.readlines():
                args = line.split(";")
                if read_continents:
                    if line == "\n":
                        read_continents = False
                        read_regions = True
                        continue
                    cont = Continent(args[1], int(args[0]), int(args[2]))
                    continent_map[cont.id] = cont
                    self.continents.append(cont)
                if read_regions:
                    if line == "\n":
                        read_regions = False
                        read_edges = True
                        continue
                    try:
                        # region = Region(Path(), args[1], int(args[0]), continent_map[int(args[2])])
                        region = Region(
                            args[1], int(args[0]), continent_map[int(args[2])]
                        )

                    except KeyError:
                        logging.error(f"not found {args[0]}")
                        raise KeyError(args[0])
                    self.regions.append(region)
                    region_map[int(args[0])] = region
                if read_edges:
                    if line == "\n":
                        continue
                    _from = int(args[0])
                    _to = int(args[1])
                    froms.append(_from)
                    tos.append(_to)
                    region_map[_from].add_neighbour(region_map[_to])
                    region_map[_to].add_neighbour(region_map[_from])
            dummy_froms = froms.copy()
            froms += tos
            tos += dummy_froms
            self.torch_edge_list.append(froms)
            self.torch_edge_list.append(tos)
        else:
            self.read_svg()
        logging.debug("read world file.")

    def get_region(self, name):
        for r in self.regions:
            if r.get_name().lower() == name.lower():
                return r

        raise ValueError("no region with name '" + name + "'")

    def get_continent(self, name) -> Continent | None:
        for c in self.continents:
            if c.get_name().lower() == name.lower():
                return c
        return None

    def num_regions(self):
        return len(self.regions)

    def num_continents(self):
        return len(self.continents)

