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


class Attila(AgentBase):

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
        

        num = 0
        for r in mine:
            if r.get_continent() == c:
                num += 1

        count = [0] * (num + 1)
        count[0] = 0
        count[1] = available

        for i in range(2, len(count)):
            count[i] = random.randint(0, available)

        count.sort()
        ret = []
        i = 0
        for r in mine:
            if r.get_continent() == c:
                n = count[i + 1] - count[i]
                if n > 0:
                    ret.append(PlaceArmies(r, n))
                i += 1

        return ret

    @override
    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        me = game.current_player()
        ret = []
    
        for _from in game.regions_owned_by(me):
            neighbours = _from.get_neighbours()
            random.shuffle(neighbours)
            to: Region|None = neighbours[0]
            i = 1
            while game.get_owner(to) != me and i < len(neighbours):
                to = neighbours[i]
                i += 1
            min = 1 if game.get_owner(to) == me else game.get_armies(to)
            max = game.get_armies(_from) - 1
            if min < max:
                ret.append(AttackTransfer(_from, to, min+random.randint(0, max - min)))

        return ret

