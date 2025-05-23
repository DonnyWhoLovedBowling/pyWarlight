from dataclasses import dataclass

@dataclass
class GameConfig:
    extra_armies: list[int] = None
    max_game_rounds: int = 5000
    num_players: int = 2
    manual_distribution:bool = False
    warlords: bool = False
    seed: int = -1
    map_name: str = "earth"

    def __post_init__(self):
        self.extra_armies = [0] * self.num_players

    def add_player(self, extra: int):
        self.num_players += 1
        self.extra_armies.append(extra)

    def get_csv(self):
        return f"{self.map_name} ; {self.max_game_rounds} ;  {self.manual_distribution} ; {self.warlords}"

    def get_csv_headers(self):
        return "map;maxRounds;manualDistribution;warlords"

