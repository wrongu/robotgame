import rg
import Queue
import random
import operator

def diagonals(loc, filter_out=None):
    filter_out = filter_out or []
    offsets = ((1, 1), (1, -1), (-1, -1), (-1, 1))
    locs = []
    for o in offsets:
        new_loc = tuple(map(operator.add, loc, o))
        if len(set(filter_out) & set(rg.loc_types(new_loc))) == 0:
            locs.append(new_loc)
    return locs

def multi_toward(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    mx = 1 if dx > 0 else -1
    my = 1 if dy > 0 else -1
    if dx == 0:
        return (start[0], start[1]+my)
    elif dy == 0:
        return (start[0]+mx, start[1])
    else:
        if abs(dy) > abs(dx):
            types = rg.loc_types((start[0], start[1]+my))
            if 'invalid' in types or 'obstacle' in types:
                return (start[0]+mx, start[1])
            else:
                return (start[0], start[1]+my)
        else:
            types = rg.loc_types((start[0]+mx, start[1]))
            if 'invalid' in types or 'obstacle' in types:
                return (start[0], start[1]+my)
            else:
                return (start[0]+mx, start[1])

def bfs_tree(bots, goal, player):
    # initialize "parent" grid of 'None' everywhere
    # (to be populated with (r, c) pairs of next-step towards goal)
    # if grid[r][c] is None at the end, that means that there is no path from (r,c) to goal
    grid = [None] * Robot.BOARD_HEIGHT
    for i in range(Robot.BOARD_HEIGHT):
        grid[i] = [None] * Robot.BOARD_WIDTH
    # initialize queue
    Q = Queue.Queue()
    Q.put(goal)
    while not Q.empty():
        fro = Q.get()
        surround = rg.locs_around(fro, filter_out=('invalid', 'obstacle'))
        random.shuffle(surround)
        for to in surround:
            if grid[to[0]][to[1]] is None and (to not in bots or bots[to].player_id == player):
                grid[to[0]][to[1]] = fro
                Q.put(to)
    return grid

class Target:
    
    def __init__(self, enemy, game, player):
        self.location = enemy.location
        self.enemy = enemy
        self.thePlayer = player
        self.bfs_grid = bfs_tree(game['robots'], enemy.location, player)
        self.corners = diagonals(enemy.location, filter_out=('invalid', 'obstacle'))
        self.attackers = [None] * len(self.corners)
        self.num_attackers = 0
        self.spin = 1 if random.random() < 0.5 else -1
    
    def add_attacker(self, ally):
        # check each corner, choose the closest one (naively)
        best = -1
        best_score = 1000
        for i in range(len(self.corners)):
            if self.attackers[i] is not None:
                continue
            score = rg.wdist(self.corners[i], ally.location)
            if score < best_score:
                best_score = score
                best = i
        self.attackers[best] = ally
        self.num_attackers += 1
    
    def get_move(self, ally):
        i = self.attackers.index(ally)
        if i >= 0:
            corner = self.corners[i]
            # first question is.. are we already in position?
            # if so, prepare for attack
            if ally.location == corner:
                # coordinated attacks: surround the enemy and attack all the locations it might
                # move to this turn (all aim CW or CCW based on self.spin)
                dx = self.enemy.location[0] - corner[0]
                dy = self.enemy.location[1] - corner[1]
                if dx*dy*self.spin > 0:
                    dy = 0
                else:
                    dx = 0
                return ['attack', (corner[0]+dx, corner[1]+dy)]
            # so we're not in the correct spot yet. but we may be super close.
            elif rg.wdist(corner, ally.location) <= 4:
                return ['move', multi_toward(ally.location, corner)]
            # a ways to go. use intelligent search.
            else:
                bfs_next = self.bfs_grid[ally.location[0]][ally.location[1]]
                # if no path, remove 'ally' as an attacker and do default behavior
                if bfs_next is None:
                    self.attackers[i] = None
                    return None
                else:
                    return ['move', bfs_next]
        else:
            print "lost track of allies. suicide time."
            return ['suicide']
    
    def full(self):
        return self.num_attackers >= len(self.corners)

class Robot:

    BOARD_WIDTH = 19
    BOARD_HEIGHT = 19
    
    def act(self, game):
        # shared information initialization
        if not hasattr(self, 'shared_updates'):
            self.shared_update = -1
        # shared info cleared at start of each frame:
        if self.shared_update < game['turn']:
            self.shared_update = game['turn']
            self.shared_targets = {}
        # begin the real decision-making parts
        bots = game['robots']
        allies = [bot for bot in bots.values() if bot.player_id == self.player_id]
        enemies = [bot for bot in bots.values() if bot.player_id != self.player_id]
        prioritized_enemies = sorted(enemies, key=(lambda bot: self.enemy_value(bot)))
        # loop through enemies in order of priority
        for en in prioritized_enemies:
            # get list of allies who are also targeting this enemy
            target = self.shared_targets.get(en.location)
            if target is None:
                # 'self' is the first bot to target here; make and store new target
                target = Target(en, game, self.player_id)
                self.shared_targets[en.location] = target
            elif target.full():
                # if 4 already on the job, move on to next enemy
                continue
            # work with current target here:
            target.add_attacker(self)
            turn = target.get_move(self)
            if turn is not None:
                return turn
        return self.guard_or_attack(game)
    
    def guard_or_attack(self, game):
        _, enemies = self.neighbors(game)
        if not enemies:
            return ['guard']
        elif len(enemies) * 10 > self.hp:
            return ['suicide']
        else:
            target = enemies[0]
            for en in enemies[1:]:
                if en.hp < target.hp:
                    target = en
            return ['attack', target.location]
    
    def enemy_value(self, enemy, weight=10):
        # low value is good. a robot that is low on health and close is the ideal target
        return enemy.hp + weight * rg.wdist(self.location, enemy.location)
