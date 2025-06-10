from dataclasses import dataclass
from src.game.GameConfig import GameConfig


@dataclass()
class AgentConfig:
    init: str
    name: str
    extra_armies: int

    def full_name(self):
        n = self.name
        if self.extra_armies > 0:
            n = f"{n} + {self.extra_armies}"
        return n


@dataclass
class Config:
    agent_configs: list[AgentConfig] = None
    timeout_millis = 60000
    visualize = False
    game_config = GameConfig()
    game_config.num_players = 0

    def __post_init__(self):

        self.agent_configs = [AgentConfig("neutral", "neutral", 0)]

    def is_human(self, player: int) -> bool:
        return self.agent_configs[player].init == "human"

    def add_agent(self, name: str):
        _id = None
        extra_armies = 0
        if "=" in name:
            ix = name.index("=")
        else:
            ix = -1

        if ix >= 0:
            _id = name[ix + 1 :]
            name = name[0 : ix + 1]
        if "+" in name:
            ix = name.index("+")
        else:
            ix = -1

        if ix >= 0:
            extra_armies = int(name[ix + 1])
            name = name[0 : ix + 1]
        if name == "me" or name == "human":
            if _id is None:
                _id = "You"
            name = "human"
        else:
            if _id is None and "." in name:
                _id = name[name.rindex(".") + 1]
            name = f"internal: {name}"
        ag = AgentConfig(name, _id, extra_armies)
        self.agent_configs.append(ag)
        self.game_config.add_player(extra_armies)

    def add_human(self):
        self.add_agent("human")

    def player_name(self, player: int):
        return self.agent_configs[player].name

    def full_name(self, player: int):
        self.agent_configs[player].full_name()

    def num_players(self):
        return len(self.agent_configs)

    def get_csv_header(self):
        sb = ""
        for p in range(1, self.num_players() + 1):
            sb += f";player {p}"
        return self.game_config.get_csv_headers() + ";" + "timeoutMillis" + sb

    def get_csv(self):
        sb = ""
        for p in range(1, self.num_players()):
            sb += f";{self.full_name(p)}"
        return self.game_config.get_csv() + ";" + str(self.timeout_millis) + sb

    def agent_init(self, i):
        return self.agent_configs[i].init
