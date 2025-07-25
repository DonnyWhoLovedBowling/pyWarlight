import logging

from line_profiler_pycharm import profile
import os
# from svgelements import (
#     SVG,
#     Group,
#     Desc,
#     Path,
#     Move,
#     Line,
#     QuadraticBezier,
#     CubicBezier,
#     Close,
#     Point,
#     SVGElement,
#     Text,
# )
# from svgpathtools import svg2paths2
from src.game.Continent import Continent
from src.game.Region import Region
# from src.utils.Util import to_global, is_point_in_path


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


# def get_child_by_name(e: SVG, name: str) -> Group:
#     for child in e:
#
#         if child.id == name:
#             return child
#     raise Exception(f"Could not find child with name: {name}")
#

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

    # def read_svg(self):
    #     world_path = "res\\maps\\earth.svg"
    #     paths, attributes, svg_attributes = svg2paths2(world_path)
    #     # print(svg_attributes)
    #     f_name = world_path
    #
    #     self.diagram = SVG.parse(f_name)
    #
    #     _map = get_child_by_name(self.diagram, "map")
    #     _map_desc = self.create_regions(_map)
    #     out_file = open("world.txt", "a")
    #     for c in self.continents:
    #         out_file.write(f"{c.id};{c.name};{c.get_reward()} \n")
    #     out_file.write("\n")
    #     for r in self.regions:
    #         out_file.write(f"{r.id};{r.name};{r.continent.id} \n")
    #     out_file.write("\n")
    #     out_file.close()
    #     self.find_neighbors()
    #
    #     if _map_desc is not None:
    #         for line in _map_desc.splitlines():
    #             line = line.strip()
    #             i = line.index(" ")
    #             keyword = line[0:i]
    #             line = line[i + 1 :]
    #             if keyword == "adj":
    #                 i = line.index(":")
    #                 if i == -1:
    #                     raise ValueError(f"expected : in {line}")
    #                 name1 = line[:i].strip()
    #                 r = self.get_region(name1)
    #                 name2 = line[i + 1 :].strip()
    #                 s = self.get_region(name2)
    #                 r.add_neighbour(s)
    #                 self.regions.append(r)
    #
    #     self.find_labels()
    #     # self.find_rewards()
    #
    # def create_regions(self, _map: Group):
    #     map_desc = None
    #     for e in _map:
    #         if type(e) == Desc:
    #             map_desc = e.desc
    #         else:
    #             c = Continent(name=e.id, id=len(self.continents))
    #             self.continents.append(c)
    #
    #             for child in e:
    #                 if type(child) == Path:
    #                     p = child
    #                     # r = Region(p, p.values['title'], len(self.regions), c)
    #                     r = Region(p.values["title"], len(self.regions), c)
    #
    #                     self.regions.append(r)
    #                     c.add_region(r)
    #                 elif type(child) == Desc:
    #                     s: str = child.desc
    #                     w = s.split(" ")
    #                     if w[0].strip() == "bonus":
    #                         c.set_reward(int(w[1]))
    #                     else:
    #                         raise ValueError("unknown keyword in continent description")
    #     return map_desc
    #
    # # def get_name(self, e: et.Element):

    def find_neighbors(self):
        dist: int = 3
        unique_neighbors = set()
        coords: list = [0.0] * 6  # Python list to store coordinates
        out_file = open("world.txt", "a")
        for r in self.regions:
            rp = r.path
            for s in self.regions:
                if r.id >= s.id:
                    continue
                sp = s.path
                if do_bounding_boxes_intersect(rp.bbox(), sp.bbox()):
                    for segment in rp:
                        i = -1
                        if isinstance(segment, Move):
                            i = 0
                            coords[0] = segment.end.x
                            coords[1] = segment.end.y
                        elif isinstance(segment, Line):
                            coords[0] = segment.end.x
                            coords[1] = segment.end.y
                            i = 0
                        elif isinstance(segment, CubicBezier):
                            coords[0] = segment.control1.x
                            coords[1] = segment.control1.y
                            coords[2] = segment.control2.x
                            coords[3] = segment.control2.y
                            coords[4] = segment.end.x
                            coords[5] = segment.end.y

                            i = 4
                        elif isinstance(segment, QuadraticBezier):
                            coords[0] = segment.control.x
                            coords[1] = segment.control.y
                            coords[2] = segment.end.x
                            coords[3] = segment.end.y

                            i = 2
                        elif isinstance(segment, Close):
                            pass
                        else:
                            print(f"Unknown segment", type(segment))
                        # if do_bounding_boxes_intersect(sp.bbox(), (coords[i] - dist, coords[i + 1] - dist, 2 * dist, 2 * dist)):
                        if (
                            do_bounding_boxes_intersect(sp.bbox(), rp.bbox())
                            and (r.id, s.id) not in unique_neighbors
                        ):

                            r.add_neighbour(s)
                            s.add_neighbour(r)
                            unique_neighbors.add((r.id, s.id))
                            out_file.write(f"{r.id};{s.id} \n")
                            # print(f"neighbor! {r.name} {s.name}")

        out_file.close()

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

    # def find_labels(self):
    #     text = get_child_by_name(self.diagram, "text")
    #     for e in text:
    #         p = Point(float(e.values["x"]), float(e.values["y"]))
    #         p = to_global(e, p)
    #         for r in self.regions:
    #             s = to_global(r.path, p)
    #             if is_point_in_path(r.path, s):
    #                 r.set_label_position(s)
    #                 break
    #         try:
    #             e.values["display"] = None
    #         except BaseException as ex:
    #             raise ValueError(ex)

    def find_rewards(self):
        rewards = get_child_by_name(self.diagram, "rewards")
        if rewards is None:
            return
        for g in rewards:
            name = g.title.upper()
            c = self.get_continent(name)
            c.reward_element = g

    def num_regions(self):
        return len(self.regions)

    def num_continents(self):
        return len(self.continents)


# if __name__ == "__main__":
#     world = World([], [])
#     world.read_svg()
