import logging
import math
import random
import time
import sys
from builtins import int
from collections import deque

from src.game.Continent import Continent

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override

from src.engine.AgentBase import AgentBase
from src.game.Game import Game
from src.game.Region import Region
from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies

class Saladin(AgentBase):

    def __init__(self):
        super().__init__()
        self.logger  = logging.getLogger(__name__)
        self.dist_from_target = dict()

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())

    def dist_from(self, game: Game, regions: list[Region]) -> int:
        dist = []
        for i in range(len(dist)):
            dist.append(-1)
        queue = deque()
        for r in regions:
            dist[r.get_id()] = 0
            queue.append(r)
        while queue:
            r = queue.popleft()
            for s in r.get_neighbours():
                if dist[s.get_id()] == -1:
                    dist[s.get_id()] = dist[r.get_id()] + 1
                    queue.append(s)

        return dist


    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        return random.choice(choosable)

    def is_target(self, game: Game, region: Region, goal: Continent ) -> bool:
        me = game.current_player()
        owner = me.get_owner(region)
        if region.get_continent() == goal and owner != me:
            return True

        if owner != me and owner != -1:
            for s in region.get_neighbours():
                if game.get_owner(s) == me:
                    c = s.get_continent()
                    if c == goal or game.get_owner(c) == me:
                        return True
        return False

    def  is_target(self, game: Game, region: Region, goal: Continent ) -> bool:
        me = game.current_player()
        owner = me.get_owner(region)
        if region.get_continent() == goal and owner != me:
            return True

        if owner != me and owner != -1:
            for s in region.get_neighbours():
                if game.get_owner(s) == me:
                    c = s.get_continent()
                    if c == goal or game.get_owner(c) == me:
                        return True
        return False

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        me = game.current_player()
        available = game.armies_per_turn(me)
        
        mine = game.regions_owned_by(me)
        dist_from_me = dict()
        for m in mine:
            dist_from_me[m.get_id()] = game.proximity_to_nearest_enemy(m)

        goal = None
        best = -1
        for c in game.world.continents:
            if game.get_owner(c) == me:
                continue
            min_dist = 1e6
            missing = 0
            for r in c.get_regions():
                min_dist = min(dist_from_me[r.id], min_dist)
                owner = game.g




















































































































                et_owner(r)
                if ower == 0:
                    missing += 1
                elif owner != me:
                    missing += 2
            score = missing + min_dist
            if score < best:
                goal = c
                best = score

        targets = []
        for r in game.world.regions:
            if self.is_target(game, r, goal):
                targets.append(r)
        self.dist_from_target = self.dist_from(game, targets)
        min_dist_from_target = 1e6
        for r in mine:
            if self.dist_from_target[r.get_id()] < min_dist_from_target:
                min_dist_from_target = self.dist_from_target[r.get_id()]
        dest = []
        for r in mine:
            if game.is_enemy_border(r) and self.dist_from_target[r.get_id()] == min_dist_from_target:
                dest.append(r)
        if len(dest) == 0:
            for r in mine:
                if self.dist_from_target[r.get_id()] == min_dist_from_target:
                    dest.append(r)
        if len(dest) == 0:
            dest = mine

        count  = []
        count.append(0)
        count.append(available)
        for i in range(2, len(dest)+1):
            count.append(random.randint(available))
        count.sort()
        ret = []
        i = 0
        for r in dest:
            n = count[i+1] - count[i]
            if n > 0:
                ret.append(PlaceArmies(r, n))
            i += 1
        return ret


    @override
    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        me = game.current_player()
        ret = []
    
        for r in game.regions_owned_by(me):
            neighbours = r.get_neighbours()
            random.shuffle(neighbours)
            to: Region|None = None
            for n in neighbours:
                if to is None or self.dist_from_target[n.get_id()] < self.dist_from_target[to.get_id()]:
                    to = n

            mn = 1 if game.get_owner(to) == me else math.ceil(game.get_armies(to) * 1.6)
            mx = game.get_armies(r) - 1

        if mn <= mx:
            ret.append(AttackTransfer(r, to, mn + random.randint(mx - mn)))
        return ret

