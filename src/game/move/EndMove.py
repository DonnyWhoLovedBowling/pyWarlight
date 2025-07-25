from dataclasses import dataclass

from torch import NoneType

from src.game.move.Move import Move


@dataclass
class EndMove(Move):
    agent: NoneType

    def apply(self, game, most_likely: bool):
        game.end_move(self.agent)
