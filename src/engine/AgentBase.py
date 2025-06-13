from src.game.Phase import Phase
from src.game.move.AttackTransfer import AttackTransfer
from src.game.move.AttackTransferMove import AttackTransferMove
from src.game.move.EndMove import EndMove
from src.game.move.Move import Move
from src.game.move.PlaceArmies import PlaceArmies
from src.game.move.PlaceArmiesMove import PlaceArmiesMove

import logging

class AgentBase:
    agent_number: int

    def is_rl_bot(self):
        return False

    def choose_region(self, game) -> Move:
        pass

    def place_armies(self, game) -> list[PlaceArmies]:
        pass

    def attack_transfer(self, game) -> list[AttackTransfer]:
        pass

    def init(self, timeout_millis: int):
        pass

    def terminate(self, game):
        pass

    def init_turn(self, game):
        if game.round < 3:
            logging.info(f"turn {game.round} started")
            for p in range(1, game.config.num_players + 1):
                logging.info(
                    f"player {p} owns {len(game.regions_owned_by(p))} regions and {game.number_of_armies_owned(p)} armies"
                )

    def get_move(self, game) -> AttackTransferMove | Move | PlaceArmiesMove | None:
        if game.phase == Phase.STARTING_REGION:
            return self.choose_region(game)
        elif game.phase == Phase.PLACE_ARMIES:
            return PlaceArmiesMove(self.place_armies(game))
        elif game.phase == Phase.ATTACK_TRANSFER:
            return AttackTransferMove(self.attack_transfer(game))
        elif game.phase == Phase.END_MOVE:
            return EndMove(self)

        else:
            raise NotImplementedError

    def compute_action_log_prob(self):
        pass

    def end_move(self, game):
        pass
