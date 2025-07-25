import datetime
import logging
from dataclasses import dataclass
import statsmodels.stats.proportion as proportion

from src.engine.Config import Config
from src.engine.Engine import Engine
from src.engine.GameResult import get_csv_header
from src.tournament.TotalResults import TotalResults


@dataclass
class WarlightFight:
    config: Config
    base_seed: int
    result_dir: str

    def __post_init__(self):
        logging.getLogger().setLevel(logging.INFO)

    def report_totals(
        self,
        res: TotalResults,
        total_time: list[int],
        total_moves: list[int],
        verbose: bool,
    ):

        num_players = self.config.num_players()
        for p in range(1, num_players):
            if p > 1:
                logging.info(", ")
            logging.info(
                f"{self.config.full_name(p)} = {res.total_victories[p]} ({100*res.total_victories[p]:.2f}%)"
            )

        if num_players > 2 and verbose:
            logging.info("average score: ")
            for p in range(1, num_players):
                if p > 1:
                    logging.info(", ")
                logging.info(
                    f"{self.config.full_name(p)} = {res.total_scores[p]/self.config.game_config.num_games}"
                )

        for p in range(1, num_players):
            if p > 1:
                logging.info(", ")
            logging.info(
                f"{self.config.full_name(p)} took {total_time[p]/total_moves[p]} ms/move"
            )

        logging.info("")

        if self.config.game_config.num_games == 1:
            return

        if verbose:
            confidence = 98
            logging.info(f"with {confidence}: confidence")
            for p in range(1, num_players + 1):
                confidence_level = confidence / 100.0
                lower_bound, upper_bound = proportion.proportion_confint(
                    res.total_victories[p],
                    self.config.game_config.num_games,
                    alpha=1 - confidence_level,
                    method="wilson",
                )

                lo = lower_bound * 100
                hi = upper_bound * 100
                logging.info(f"  {self.config.full_name(p)} wins {lo:.1f}% - {hi:.1f}%")

        if self.result_dir is None:
            return

        try:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_file = open(f"total_results_{ts}.csv", "w")
            out_file.write(f"datetime;{self.config.get_csv_header()} \n")
            out_file.write(
                f"{datetime.datetime.now()};{self.config.get_csv()};{self.base_seed};{res.get_csv()}"
            )
        except BaseException:
            logging.exception("failed to write matches.csv")

    def fight(self, verbose: bool) -> TotalResults:
        num_players = self.config.num_players()
        total_results: TotalResults = TotalResults(num_players, 100)
        total_moves = [0] * (num_players + 1)
        total_time = [0.0] * (num_players + 1)
        out_file = None

        if self.result_dir is not None:
            out_file = open("matches.csv", "a")
            out_file.write(
                f"datetime;{self.config.get_csv_header()};seed;{get_csv_header(num_players)} \n"
            )
        engine = Engine(self.config)
        if len(engine.World.regions) < 10:
            self.config.agent_configs = self.config.agent_configs[-1:]
        for game in range(self.config.game_config.num_games):
            logging.info(f"starting game {game}")
            seed = self.base_seed + game
            self.config.game_config.seed = seed
            result = engine.run(verbose=verbose)
            for p in range(1, num_players + 1):
                total_moves[p] += result.total_moves[p]
                total_time[p] += result.total_time[p]

            logging.info(
                f"seed {seed}: {self.config.agent_configs[result.winner]} won in {result.round} rounds"
            )
            if verbose and num_players > 2:
                logging.info(" (")
                for i in range(2, num_players + 1):
                    if i > 2:
                        logging.info(", ")
                    p = 1
                    while True:
                        if result.score[p] == num_players - i:
                            break
                        p += 1
                    logging.info(f"{i} = {self.config.full_name(p)}")
                logging.info(")")
            logging.info("")
            if result.winner != -1:
                total_results.total_victories[result.winner] += 1
            for p in range(1, num_players):
                total_results.total_scores[p] += result.score[p]

            if out_file is not None:
                out_file.write(
                    f"{datetime.datetime.now()};{self.config.get_csv()}; {seed}; {result.get_csv()}\n"
                )
            if game % 100 == 0:
                self.report_totals(total_results, total_time, total_moves, verbose)
                total_results: TotalResults = TotalResults(num_players, 100)
        if out_file is not None:
            out_file.close()

        return total_results
