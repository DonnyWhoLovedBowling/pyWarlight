from src.game.Game import Game
from src.game.Phase import Phase
from src.game.Region import Region
from src.game.Continent import Continent
from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.AttackTransferMove import AttackTransferMove
from src.game.move.Move import Move
from src.game.move.PlaceArmies import PlaceArmies
from src.game.move.PlaceArmiesMove import PlaceArmiesMove


class AgentBase:


    def choose_region(self, game: Game) -> Move:
        pass

    def place_armies(self, game: Game) -> list[PlaceArmies]:
        pass

    def attack_transfer(self, game: Game) -> list[AttackTransfer]:
        pass

    def init(self, timeout_millis: int):
        pass

    def terminate(self):
        pass

    def get_move(self, game: Game) -> Move:
        if game.phase == Phase.STARTING_REGION:
            return self.choose_region(game)
        elif game.phase == Phase.PLACE_ARMIES:
            return PlaceArmiesMove(self.place_armies(game))
        elif game.phase == Phase.ATTACK_TRANSFER:
            return AttackTransferMove(self.attack_transfer(game))
        else:
            raise NotImplementedError

