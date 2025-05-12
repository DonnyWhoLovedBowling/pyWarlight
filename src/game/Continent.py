import logging
from dataclasses import  dataclass, field

from svgelements import Group



@dataclass
class Continent:
    name: str
    id: int
    reward: int = 0
    regions: list = field(default_factory= lambda: [])

    # reward_element = Group

    def get_id(self):
        return self.id

    def get_regions(self) -> list:
        return self.regions

    def get_name(self):
        return self.name

    def get_reward(self):
        return self.reward

    def set_reward(self, reward):
        self.reward = reward

    def add_region(self, region):
        if region.continent.id != self.id:
            logging.error("wrong region added!")
            raise ValueError
        self.regions.append(region)

