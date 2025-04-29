from dataclasses import dataclass

from src.game.Game import Game
from src.game.move.Move import Move
from src.game.move.PlaceArmies import PlaceArmies

@dataclass
class PlaceArmiesMove(Move):
    commands: list[PlaceArmies]

    def __eq__(self, other):
        if not isinstance(other, PlaceArmiesMove):
            return False

        return self.commands == other.commands

    def apply(self, state: Game, most_likely: bool):
        state.place_armies(self.commands)

    def __str__(self):
        sb = 'PlaceArmiesMove('
        for i in range(len(self.commands)):
            if i > 0:
                sb += ', '
            p = self.commands[i]
            sb += p.region.name + ' = ' + str(p.armies)

        sb += ')'
        return sb

