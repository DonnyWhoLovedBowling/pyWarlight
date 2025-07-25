import logging
import math
import random
import time
import sys
from builtins import int

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override

from src.engine.AgentBase import AgentBase
from src.game.Game import Game
from src.game.Region import Region
from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.PlaceArmies import PlaceArmies


def priority(game: Game, from_region: Region, to_region: Region) -> int:
    me = game.current_player()
    who = game.get_owner(to_region)
    if who == me:
        return 1
    elif who == 0:
        return 3 if to_region.get_continent() == from_region.get_continent() else 2
    else: 
        return 4


class Napoleon(AgentBase):

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())

    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        return random.choice(choosable)

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        me = game.current_player()
        available = game.armies_per_turn(me)
        
        mine = game.regions_owned_by(me)
        c = mine[0].get_continent()
        for r in mine:
            c1 = r.get_continent()
            if (game.get_owner(c1) != me and
                (game.get_owner(c) == me or c1.get_reward() < c.get_reward())):
                c = c1
        

        dest = []
        for r in mine:
            c1 = r.get_continent()
            if game.is_enemy_border(r) and (c1 == c or game.get_owner(c1) == me):
                dest.append(r)

        if len(dest) == 0:
            for r in mine:
                if r.get_continent() == c and game.is_border(r):
                    dest.append(r)

        count = [0] * (len(dest) + 1)
        count[1] = available
        for i in range(2, len(dest)):
            count[i] = random.randint(0, available)
        count.sort()
        ret = []
        for i, r in enumerate(dest):
            n = count[i + 1] - count[i]
            if n > 0:
                ret.append(PlaceArmies(r, n))

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
                if to is None or priority(game, r, n) > priority(game, r, to):
                    to = n

            mn = 1 if game.get_owner(to) == me else math.ceil(game.get_armies(to) * 1.5)
            mx = game.get_armies(r) - 1

            if mn <= mx:
                ret.append(AttackTransfer(r, to, mx))

        return ret

