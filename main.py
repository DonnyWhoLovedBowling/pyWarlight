from src.engine.AgentBase import AgentBase
from src.engine.Config import Config
from src.engine.Engine import Engine
from src.tournament.WarlightFight import WarlightFight

import sys

def simulate_games(config: Config, agents: list[str], seed: int,
                   games: int, result_dir: str, verbose: bool):

    if len(agents) < 2:
        raise Exception("Must have at least 2 agents")

    config.visualize = False
    for a in agents:
        config.add_agent(a)
    fight = WarlightFight(config, max(seed, 0), games, result_dir)
    fight.fight(verbose)

if __name__ == "__main__":
    agents = []
    seed = -1
    sim = 0
    verbose = False
    result_dir = 'results'
    config = Config()
    skip_next = False
    for i in range(1, len(sys.argv)):
        if skip_next:
            skip_next = False
            continue

        s = sys.argv[i]
        if s.startswith("-"):
            if s == '-manual':
                config.game_config.manual_distribution = True
            elif s == '-map':
                config.game_config.map_name = sys.argv[i + 1]
                skip_next = True
            elif s == "-maxrounds":
                config.game_config.max_rounds = int(sys.argv[i + 1])
            elif s == "-resultdir":
                config.game_config.result_dir = sys.argv[i + 1]
                skip_next = True
            elif s == "-verbose":
                config.game_config.verbose = True
            elif s == "-seed":
                seed = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-sim":
                sim = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-timeout":
                config.game_config.timeout = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-warlords":
                config.game_config.warlords = True
            else:
                raise Exception("Invalid argument", s)
        else:
            agents.append(sys.argv[i])
    if len(agents) > 4:
        raise Exception("Too many agents")
    if sim > 0:
        simulate_games(config, agents, seed, sim, result_dir, verbose)
    else:
        if len(agents) < 2:
            raise Exception("Must have at least 2 agents")
        else:
            for s in agents:
                config.add_agent(s)

    config.visualize = False
    config.game_config.seed = seed
    Engine(config).run(verbose)

    exit(0)
