from dataclasses import  dataclass

from svgelements import Group



@dataclass
class Continent:
    name: str
    id: int
    reward: int = 0
    regions = []
    reward_element = Group

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
        self.regions.append(region)

