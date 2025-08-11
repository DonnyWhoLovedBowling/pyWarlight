from src.engine.Config import Config
from src.engine.Engine import Engine
from src.tournament.WarlightFight import WarlightFight

import sys


def simulate_games(_config: Config, _agents: list[str], _seed: int,
                   _result_dir: str, _verbose: bool):

    if len(_agents) < 2:
        raise Exception("Must have at least 2 agents")

    _config.visualize = False
    for a in _agents:
        _config.add_agent(a)
    fight = WarlightFight(_config, max(_seed, 0), _result_dir)
    fight.fight(_verbose)

if __name__ == "__main__":
    agents = []
    seed = -1
    sim = 0
    verbose = False
    result_dir = 'results'
    config = Config()
    skip_next = False
    rlgnn_config = ''  # Store the RLGNNAgent config name

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
                config.game_config.num_players = int(sys.argv[i + 1])
            elif s == "-resultdir":
                config.game_config.result_dir = sys.argv[i + 1]
                skip_next = True
            elif s == "-verbose":
                config.game_config.verbose = True
            elif s == "-seed":
                seed = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-sim":
                config.game_config.num_games = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-timeout":
                config.game_config.timeout = int(sys.argv[i + 1])
                skip_next = True
            elif s == "-warlords":
                config.game_config.warlords = True
            elif s == "-config":
                config.game_config.rlgnn_config = sys.argv[i + 1]
                skip_next = True
            else:
                raise Exception("Invalid argument", s)
        else:
            agents.append(sys.argv[i])
        
    if len(agents) > 4:
        raise Exception("Too many agents")
    if config.game_config.num_games > 0:
        simulate_games(config, agents, seed, result_dir, verbose)
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
