from src.game.Game import Game
from src.game.move.Move import Move
from src.game.move.AttackTransfer import AttackTransfer

from dataclasses import dataclass

@dataclass
class AttackTransferMove(Move):
    commands: list[AttackTransfer]

    def __eq__(self, other):
        return self.commands == other.commands

    def apply(self, game: Game, mostLikely: bool):
        game.attack_transfer(self.commands, mostLikely)

    def __str__(self):
        return ", ".join(map(str, self.commands))