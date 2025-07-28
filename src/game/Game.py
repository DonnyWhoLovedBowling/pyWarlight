import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field

from src.engine.AgentBase import AgentBase
from src.game.Continent import Continent
from src.game.FightSide import FightSide
from src.game.move.FigthResult import FightResult
from src.game.move.PlaceArmies import PlaceArmies
from src.game.Region import Region
from src.game.GameConfig import GameConfig
from src.game.World import World
from src.game.Phase import Phase
import random, datetime
from datetime import datetime
from multipledispatch import dispatch

from src.game.move.Move import Move

from src.game.move.AttackTransfer import AttackTransfer


def manual_round(d: float, most_likely: bool) -> int:

    if most_likely:
        return round(d)
    p = d - math.floor(d)
    return int(math.ceil(d) if random.random() < p else math.floor(d))


@dataclass
class \
        Game:
    config: GameConfig
    world: World
    armies: list[int] = field(default_factory=lambda: [])
    owner: list[int] = field(default_factory=lambda: [])
    round: int = 0
    turn: int = 0
    phase: Phase = field(default_factory=lambda: [])
    score: list[int] = field(default_factory=lambda: [])
    pickable_regions: list[Region] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.armies = [2] * self.world.num_regions()
        self.owner = [-1] * self.world.num_regions()
        self.score = [-1] * (self.config.num_players + 1)
        self.turn = 1
        random.seed(datetime.now().timestamp())
        self.init_starting_regions()

    def max_score(self):
        _max = -1
        for p in range(self.config.num_players):
            _max = max(_max, self.score[p])
        return _max

    @dispatch(Region)
    def get_owner(self, region: Region):
        return self.owner[region.get_id()]

    @dispatch(int)
    def get_owner(self, region: int):
        return self.owner[region]


    @dispatch(Continent)
    def get_owner(self, continent: Continent):
        player = self.get_owner(continent.get_regions()[0])
        for region in continent.get_regions():
            if player != self.get_owner(region):
                return 0
        return player

    def set_owner(self, region: Region, player: int) -> None:
        old = self.owner[region.get_id()]
        self.owner[region.get_id()] = player

        if old > 0 and self.number_of_regions_owned(old) == 0:
            _max = self.max_score()
            if self.score[old] != -1:
                raise ValueError("score should be -1")
            self.score[old] = _max + 1

            if self.score[old] == self.config.num_players - 2:
                for p in range(1, self.config.num_players + 1):
                    if self.score[p] == -1:
                        self.score[p] = self.config.num_players - 1
                        break

    def get_armies(self, region: Region) -> int:
        return self.armies[region.id]

    def set_armies(self, r: Region, n: int) -> None:
        self.armies[r.get_id()] = n
        if n == 0:
            raise ValueError(f"setting armies for region {r.name} to {n}")

    def number_of_regions_owned(self, player: int) -> int:
        n = 0
        for r in self.world.regions:
            if self.get_owner(r) == player:
                n += 1
        return n

    def number_of_armies_owned(self, player: int) -> int:
        n = 0
        for r in self.world.regions:
            if self.get_owner(r) == player:
                n += self.get_armies(r)
        return n

    def is_owned_by(self, region: Region, player: int) -> bool:
        return self.owner[region.get_id()] == player

    def regions_owned_by(self, player: int) -> list[Region]:
        owned_regions = []
        for region in self.world.regions:
            owner = self.get_owner(region)
            if owner == player:
                owned_regions.append(region)
        return owned_regions

    def winning_player(self) -> int:
        for p in range(1, self.config.num_players + 1):
            if self.score[p] == self.config.num_players - 1:
                return p
        logging.warning("No winning player")
        return -1

    def is_done(self) -> bool:
        for p in range(1, self.config.num_players + 1):
            logging.debug(f"score {p}: {self.score[p]}")
            if self.score[p] == -1:
                return False
        return True

    @dispatch(int, bool)
    def armies_per_turn(self, player: int, first: bool) -> int:
        armies = 5
        if first and self.config.num_players == 2:
            armies = 3
        armies += self.get_bonus_armies(player)
        return armies

    @dispatch(int)
    def armies_per_turn(self, player: int) -> int:
        return self.armies_per_turn(player, player == 1 and self.round <= 1)

    @dispatch(int)
    def armies_each_turn(self, player: int) -> int:
        return self.armies_per_turn(player, False)

    def num_starting_regions(self):
        n_regions = 0
        if self.config.warlords:
            n_regions = (self.world.num_continents() / self.config.num_players)
        elif len(self.world.regions) < 5:
            n_regions = 1
        elif len(self.world.regions) <= 12:
            n_regions = 2
        else:
            n_regions = 4
        return n_regions

    def borders_enemy(self, region: Region, player: int) -> bool:
        for n in region.get_neighbours():
            owner = self.get_owner(n)
            if owner != player and owner != -1:
                return True
        return False

    def regions_on_continent(self, continent: Continent, player: int) -> int:
        count = 0
        for s in continent.get_regions():
            if s.continent.id != continent.id:
                continue
            if self.get_owner(s) == player:
                count += 1
        return count

    def get_random_starting_region(self, player) -> Region:
        for p in range(1, 3):
            possible = []
            for r in self.pickable_regions:
                # print(f'checking if {r.get_id()} is pickable')
                if (self.regions_on_continent(r.get_continent(), player) < 2 and (
                    (not self.borders_enemy(r, player)) or p == 2
                )) or len(self.world.regions) < 12:
                    possible.append(r)

            if len(possible) > 0:
                # print('possible!', possible)
                return random.choice(possible)
        raise Exception("No possible starting region")

    def set_as_starting(self, r: Region, player: int) -> None:
        self.set_owner(r, player)
        self.set_armies(r, 2 + self.config.extra_armies[player])
        self.pickable_regions.remove(r)

    def regions_chosen(self):
        self.round = 1
        self.phase = Phase.PLACE_ARMIES

    def init_starting_regions(self):
        self.pickable_regions = []
        if self.config.warlords:
            for c in self.world.continents:
                num_regions = len(c.get_regions())
                while True:
                    random_region_id = random.randint(0, num_regions - 1)
                    region = c.get_regions()[random_region_id]
                    ok = True
                    for n in region.get_neighbours():
                        if n in self.pickable_regions:
                            ok = False
                            break
                    if ok:
                        self.pickable_regions.append(region)
                        break
        else:
            self.pickable_regions = deepcopy(self.world.regions)

        n_starting = self.num_starting_regions()
        for i in range(n_starting):
            for p in range(1, self.config.num_players + 1):
                r = self.get_random_starting_region(p)
                self.set_as_starting(r, p)
        self.round = 1
        self.phase = Phase.PLACE_ARMIES

    def next_turn(self):
        while True:
            self.turn += 1
            logging.debug(f"doing turn {self.turn}, round: {self.round}")
            if self.turn > self.config.num_players:
                self.turn = 1
                self.round += 1

                if self.round == self.config.max_game_rounds:
                    _max = self.max_score()
                    while True:
                        min_player, min_regions, min_armies = -1, 0, 0
                        for p in range(1, self.config.num_players + 1):
                            if self.score[p] == -1:
                                r = self.number_of_regions_owned(p)
                                a = self.number_of_armies_owned(p)
                                if (
                                    min_player == -1
                                    or r < min_regions
                                    or (r == min_regions and a < min_armies)
                                ):
                                    min_player = p
                                    min_regions = r
                                    min_armies = a
                        if min_player == -1:
                            return

                        self.score[min_player] = _max + 1
            if self.score[self.turn] < 0:
                break

    def choose_region(self, r: Region) -> None:
        if self.phase != Phase.STARTING_REGION:
            raise Exception("cannot choose regions after game has begun")

        if r not in self.pickable_regions:
            raise Exception("starting region is not pickable")

        self.set_as_starting(r, self.turn)
        self.turn += 1
        if self.turn > self.config.num_players:
            self.turn = 1

        if self.number_of_regions_owned(self.turn) == self.num_starting_regions():
            self.regions_chosen()

    def illegal_move(self, s: str):
        raise ValueError(f"ignoring illegal move by player {self.turn}: {s} ")

    def place_armies(self, moves: list[PlaceArmies]) -> None:
        valid = []
        if self.phase != Phase.PLACE_ARMIES:
            self.illegal_move(f"wrong time to place armies {self.phase}")
        left = self.armies_per_turn(self.turn)
        for move in moves:
            region = move.get_region()
            armies = move.get_armies()

            if not self.is_owned_by(region, self.turn):
                self.illegal_move(
                    f"can't place armies on unowned region {region.get_name()}"
                )
            elif armies < 1:
                self.illegal_move("cannot place less than 1 army")
            elif left <= 0:
                self.illegal_move(
                    f"no {armies} armies left to place {left} {self.turn}"
                )
            else:
                if armies > left:
                    self.illegal_move(
                        f"move wants to place {armies} armies, but only {left} are available"
                    )
                self.set_armies(region, self.get_armies(region) + armies)
                left -= armies
                valid.append(move)
        self.phase = Phase.ATTACK_TRANSFER

    @dispatch(int, int, bool)
    def do_attack(
        self, attacking_armies: int, defending_armies: int, most_likely: bool
    ) -> FightResult:
        result = FightResult()
        result.defenders_destroyed = min(
            manual_round(attacking_armies * 0.6, most_likely), defending_armies
        )
        result.attackers_destroyed = min(
            manual_round(defending_armies * 0.7, most_likely), attacking_armies
        )

        result.post_process(attacking_armies, defending_armies)
        logging.debug(f"agent {self.turn} attacked")
        logging.debug(
            f"attacked, defenders destroyed: {result.defenders_destroyed}, attackers: {result.attackers_destroyed}"
        )
        logging.debug(f"defenders: {defending_armies}, attackers: {attacking_armies}")

        logging.debug(f"{result.winner} won!")

        return result

    @dispatch(AttackTransfer, bool)
    def do_attack(self, move: AttackTransfer, most_likely: bool) -> FightResult:
        from_region = move.get_from_region()
        to_region = move.get_to_region()
        attacking_armies = move.get_armies()
        defending_armies = self.get_armies(to_region)
        result = self.do_attack(attacking_armies, defending_armies, most_likely)
        if result.winner == FightSide.ATTACKER:
            self.set_armies(
                from_region, self.get_armies(from_region) - attacking_armies
            )
            self.set_owner(to_region, self.turn)
            self.set_armies(to_region, attacking_armies - result.attackers_destroyed)
        elif result.winner == FightSide.DEFENDER:
            self.set_armies(
                from_region, self.get_armies(from_region) - result.attackers_destroyed
            )
            self.set_armies(
                to_region, self.get_armies(to_region) - result.defenders_destroyed
            )
        else:
            raise Exception("Unhandled FightResult.winner: " + result.winner)
        return result

    def validate_attack_transfers(
        self, moves: list[AttackTransfer]
    ) -> list[AttackTransfer]:
        valid = []
        total_from = [0] * self.world.num_regions()
        for i in range(len(moves)):
            m = moves[i]
            from_region = m.get_from_region()
            to_region = m.get_to_region()
            if not self.is_owned_by(from_region, self.turn):
                self.illegal_move("attack/transfer from unowned region")
            elif to_region not in from_region.get_neighbours():
                self.illegal_move(
                    f"attack/transfer from {from_region.name} to region {to_region.name} that is not a neighbor"
                )
            elif m.get_armies() < 1:
                self.illegal_move("attack/transfer cannot use less than 1 army")
            elif total_from[from_region.get_id()] + m.get_armies() >= self.get_armies(
                from_region
            ):
                self.illegal_move(
                    f"attack/transfer requests more armies ({total_from[from_region.get_id()]} + "
                    f"{m.get_armies()}) than are available ({self.get_armies(from_region)})"
                )
            else:
                ok = True
                for j in range(0, i):
                    n: AttackTransfer = moves[j]
                    if (
                        n.get_from_region() == m.get_from_region()
                        and n.get_to_region() == m.get_from_region()
                    ):
                        self.illegal_move(
                            "player has already moved between same regions in this turn"
                        )
                        ok = False
                        break
                if ok:
                    total_from[from_region.get_id()] += m.get_armies()
                    valid.append(m)
        return valid

    def attack_transfer(self, moves: list[AttackTransfer], most_likely: bool):
        if self.phase != Phase.ATTACK_TRANSFER:
            self.illegal_move("wrong time to attack/transfer")
            return
        _valid = self.validate_attack_transfers(moves)

        for _move in _valid:
            from_region = _move.get_from_region()
            to_region = _move.get_to_region()
            _move.set_armies(min(_move.get_armies(), self.get_armies(from_region) - 1))
            if self.is_owned_by(to_region, self.turn):
                if _move.get_armies() == self.get_armies(from_region):
                    raise ValueError("moving all armies!")
                self.set_armies(
                    from_region, self.get_armies(from_region) - _move.get_armies()
                )
                self.set_armies(
                    to_region, self.get_armies(to_region) + _move.get_armies()
                )
            else:
                _move.result = self.do_attack(_move, most_likely)
                if self.is_done():
                    return
        self.next_turn()
        self.phase = Phase.END_MOVE

    @dispatch(Move, bool)
    def move(self, _move: Move, most_likely: bool) -> None:
        _move.apply(self, most_likely)

    @dispatch(Move)
    def move(self, _move: Move) -> None:
        self.move(_move, False)

    def pass_turn(self) -> None:
        if self.phase == Phase.STARTING_REGION:
            r = random.choice(self.pickable_regions)
            self.choose_region(r)
        elif self.phase == Phase.PLACE_ARMIES:
            owned = self.regions_owned_by(self.turn)
            r = random.choice(owned)
            place = PlaceArmies(r, self.armies_per_turn(self.turn))
            self.place_armies([place])
        elif self.phase == Phase.ATTACK_TRANSFER:
            self.attack_transfer([], False)

    def create_node_features(self) -> list[list[int]]:
        graph = []
        all_armies = sum(self.armies)
        for r in self.world.regions:
            own_armies = self.get_armies(r)
            x_list = [0] * (self.config.num_players + 6)
            owner = self.get_owner(r)
            if owner == -1:
                owner_ix = 2
            else:
                owner_ix = owner - 1
            x_list[owner_ix] = 1
            x_list[-5] = own_armies/all_armies
            allied_armies = 0
            enemy_armies = 0
            for n in r.get_neighbours():
                if self.is_enemy_border(n):
                    enemy_armies += self.get_armies(n)
                else:
                    allied_armies += self.get_armies(n)

            x_list[-4] = self.proximity_to_nearest_enemy(r)
            x_list[-3] = enemy_armies/sum(self.armies)
            x_list[-2] = allied_armies/sum(self.armies)
            x_list[-1] = own_armies
            graph.append(x_list)
        return graph

    def create_action_edges(self) -> list[list[int]]:
        action_edges = []
        for src in self.regions_owned_by(self.turn):
            if self.get_armies(src) > 1:
                for tgt in src.get_neighbours():
                    action_edges.append([src.get_id(), tgt.get_id()])
                action_edges.append([src.get_id(), src.get_id()])
        return action_edges

    def end_move(self, agent: AgentBase):
        if self.phase != Phase.END_MOVE:
            raise ValueError("not the right moment to end move!")
        agent.end_move(self)
        self.phase = Phase.PLACE_ARMIES

    def get_bonus_armies(self, player):
        armies = 0
        for cd in self.world.continents:
            if self.get_owner(cd) == player:
                armies += cd.reward
        return armies

    def current_player(self):
        return self.turn

    def is_border(self, r: Region) -> bool:
        me = self.current_player()
        for n in r.get_neighbours():
            if self.get_owner(n) != me:
                return True

        return False

    def is_enemy_border(self, r: Region) -> bool:
        me = self.current_player()
        for n in r.get_neighbours():
            if self.get_owner(n) != me and self.get_owner(n) != 0:
                return True

        return False

    def proximity_to_nearest_enemy(self, region: Region) -> int | None:
        """
        Returns the minimum number of steps from `region` to the nearest enemy region.
        If no enemy is found, returns None.
        """
        from collections import deque
        player = self.get_owner(region)
        visited = set()
        queue = deque()
        queue.append((region, 0))
        visited.add(region.get_id())

        while queue:
            current_region, distance = queue.popleft()
            # Check if current region is owned by an enemy
            owner = self.get_owner(current_region)
            if owner != player and owner != -1:
                return distance
            # Add neighbors to queue
            for neighbor in current_region.get_neighbours():
                nid = neighbor.get_id()
                if nid not in visited:
                    visited.add(nid)
                    queue.append((neighbor, distance + 1))
        return 0  # No enemy found
