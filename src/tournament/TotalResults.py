from dataclasses import dataclass

@dataclass

class TotalResults:
    num_players: int
    games: int

    def __post_init__(self):
        self.total_victories = [0] * (self.num_players + 1)
        self.total_scores = [0] * (self.num_players + 1)

    def get_csv_header(self):
        sb = "games"
        for p in range(1, self.num_players + 1):
            sb += f";wonBy{p};winRate{p}"
            if self.num_players > 2:
                sb += f";avgScore{p}"
        return sb

    def get_csv(self):
        sb = ""
        sb += "games"
        for p in range(1, self.num_players + 1):
            sb += f";{self.total_victories[p]};{round(self.total_victories[p]/self.games, 2)}"
            if self.num_players > 2:
                sb += f";{self.total_scores[p]};{round(self.total_scores[p] / self.games, 2)}"
        return sb





