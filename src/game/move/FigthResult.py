from dataclasses import dataclass

from src.game.FightSide import FightSide


@dataclass
class FightResult:
    attackers_destroyed: int = 0
    defenders_destroyed: int = 0
    winner: FightSide = None

    def post_process(self, attacking_armies: int, defending_armies: int):
        if (
            self.attackers_destroyed == attacking_armies
            and self.defenders_destroyed == defending_armies
        ):
            self.defenders_destroyed -= 1
        self.winner = (
            FightSide.ATTACKER
            if self.defenders_destroyed >= defending_armies
            else FightSide.DEFENDER
        )
