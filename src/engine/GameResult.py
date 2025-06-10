from dataclasses import dataclass

from src.engine.Config import Config
from src.game.Game import Game


def get_csv_header(num_players: int) -> str:
    sb = ""
    for p in range(1, num_players + 1):
        sb += f"p{p}Score;"
    sb += "rounds"
    for p in range(1, num_players + 1):
        sb += f";p{p}Regions;p{p}Armies"
    return sb


@dataclass
class GameResult:
    def __init__(
        self, config: Config, game: Game, total_moves: list[int], total_time: list[int]
    ) -> None:

        self.config: Config = config
        self.total_moves: list[int] = total_moves
        self.total_time: list[int] = total_time
        self.regions: list[int] = [0] * (config.num_players() + 1)
        self.armies: list[int] = [0] * (config.num_players() + 1)
        for p in range(1, config.num_players() + 1):
            self.regions[p] = game.number_of_regions_owned(p)
            self.armies[p] = game.number_of_armies_owned(p)

        self.winner: int = game.winning_player()
        self.score: list[int] = game.score
        self.round: int = game.round

    def get_csv(self):
        sb = ""
        for p in range(1, self.config.num_players()):
            sb += f"{self.score[p]};"
        sb += f"{self.round}"
        for p in range(1, self.config.num_players()):
            sb += f";{self.regions[p]};{self.armies[p]};"
        return sb
