import logging
import random
import time
import sys

if sys.version_info[1] < 11:
    from typing_extensions import override
else:
    from typing import override

from src.engine.AgentBase import AgentBase
from src.game.Game import Game
from src.game.Region import Region
from src.game.move.AttackTransfer import AttackTransfer
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
        count = [0] * num_regions
        if num_regions == 0:
            logging.error(f"agent {me} has no regions!")
            raise ValueError
        for i in range(available):
            r = random.randint(0, num_regions - 1)
            count[r] += 1
        ret = []
        for i in range(num_regions):
            if count[i] > 0:
                ret.append(PlaceArmies(mine[i], count[i]))
        return ret

    def attack_transfer(self, game: Game) -> list[AttackTransfer]:

        me = game.turn
        ret = []

        for r in game.regions_owned_by(me):
            regional_armies = game.get_armies(r)
            if regional_armies > 0:
                count = random.randrange(0, regional_armies)
            else:
                raise ValueError(f"No armies on region {r.name}")
            if count > 0:
                neighbors = r.get_neighbours()
                to = random.choice(list(neighbors))
                ret.append(AttackTransfer(r, to, count, None))
        if game.round % 50 == 1:
            logging.info(
                f"at round {game.round}, random agent does {len(ret)} attack/transfers"
            )
            if len(ret) > 0:
                logging.info(
                    f"using {sum([a.armies for a in ret])/len(ret)} armies on avg"
                )

        return ret

    def terminate(self, game: Game):
        logging.info("agent terminated")
        self.end_move(game)
