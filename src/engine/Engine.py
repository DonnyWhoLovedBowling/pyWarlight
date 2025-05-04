from copy import deepcopy
from logging import error, debug
from time import time

from src.agents.RandomAgent import RandomAgent
from src.engine.AgentBase import AgentBase
from src.engine.Config import Config
from src.engine.GameResult import GameResult
from src.engine.HumanAgent import HumanAgent
from src.game.Game import Game


def construct_agent(agent_fqcn: str) -> AgentBase:
    if 'random' in agent_fqcn:
        return RandomAgent()
    else:
        return HumanAgent(agent_fqcn)


class Engine:

    def __init__(self, config: Config):
        self.config: Config = config
        self.game: Game = Game(config.game_config)
        self.agents: list[AgentBase] = None

    def timeout(self, agent: AgentBase, elapsed: int):
        if not isinstance(agent, HumanAgent) and self.config.timeout_millis > 0 and elapsed > config.timeout_millis + 150:
            error(f"agent failed to respond in time!  timeout = {self.config.timeout_millis}, elapsed = {elapsed}")
            return True
        return False

    def setup_agent(self, agent_init: str) -> AgentBase:
        if agent_init.startswith('internal:'):
            agent_fqcn = agent_init[len('internal:'):]
            ret = construct_agent(agent_fqcn)
        elif agent_init.startswith('human:'):
            self.config.visualize = True
            ret = HumanAgent()
        else:
            raise Exception(f"unknown agent type: {agent_init}")
        return ret

    def run(self, verbose: bool = False) -> GameResult:
        game = Game(self.config.game_config)
        players =  self.config.num_players()
        agents = [AgentBase] * players + 1
        for p in range(1, players+1):
            agents[p] = self.setup_agent(self.config.agent_init[p])
            agents[p].init(self.config.timeout_millis)

        total_moves = [0] * players + 1
        total_time = [0.] * players + 1
        round = -1

        while not self.game.is_done():
            if verbose and game.round != round:
                round = game.round
                debug(f"Round {round}")
            player = game.turn
            agent = agents[player]

            start = time()
            move = agent.get_move(deepcopy(game))
            elapsed = time() - start
            total_moves[player] += 1
            total_time[player] += elapsed

            if self.timeout(agent, int(elapsed)):
                self.game.pass_turn()
            else:
                self.game.move(move)


        if verbose:
            debug("\r")
        for p in range(1, players+1):
            agents[p].terminate()
        return GameResult(self.config, total_moves, total_time)





