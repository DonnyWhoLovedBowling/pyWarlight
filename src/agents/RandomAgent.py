import random
import time
from typing import override

from src.engine.AgentBase import AgentBase
from src.game.Game import Game
from src.game.Region import Region
from src.game.move.Move import Move
from src.game.move.PlaceArmies import PlaceArmies


class RandomAgent(AgentBase):

    @override
    def init(self, timeout_millis: int):
        random.seed(time.time())

    @override
    def choose_region(self, game: Game) -> Region:
        choosable = game.pickable_regions
        return random.choice(choosable)

    @override
    def place_armies(self, game: Game) -> list[PlaceArmies]:
        me = game.turn
        available = game.armies_per_turn(me)
        mine = game.regions_owned_by(me)
        num_regions = len(mine)
        count = [] * num_regions
        for i in range(available):
            r = random.randint(0, num_regions - 1)
            count[r] += 1
        ret = []
        for i in range(num_regions):
            if count[i] > 0:
                ret.append(PlaceArmies(mine[i], count[i]))
        return ret



