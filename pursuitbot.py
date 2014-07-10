import rg
import Queue
import random

def dfs_group(bots, loc, player, partial_group, diagonal=True):
    if diagonal:
        for i in range(loc[0]-1, loc[0]+2):
            for j in range(loc[1]-1, loc[1]+2):
                pt = (i,j)
                # if pt is on the board
                # and there is a bot at pt
                # and that bot is of the same type
                # and we haven't already added that bot to the group
                if 'normal' in rg.loc_types(pt) and pt in bots and bots[pt].player_id == player and pt not in partial_group:
                    # add it to the group and recurse
                    partial_group.append((i,j))
                    partial_group = dfs_group(bots, (i,j), player, partial_group, diagonal)
    else:
        for pt in rg.locs_around(loc, filter_out=('invalid', 'obstacle')):
            if 'normal' in rg.loc_types(pt) and pt in bots and bots[pt].player_id == player and pt not in partial_group:
                # add it to the group and recurse
                partial_group.append((i,j))
                partial_group = dfs_group(bots, (i,j), player, partial_group, diagonal)
    return partial_group
    
def print_grid(grid):
    print "========================================"
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            cell = grid[i][j]
            if cell is None:
                if 'obstacle' in rg.loc_types((i,j)):
                    print "#",
                else:
                    print ".",
            else:
                if cell[0] > i:
                    print "V",
                elif cell[0] < i:
                    print "^",
                elif cell[1] > j:
                    print ">",
                elif cell[1] < j:
                    print "<",
                else:
                    print "?",
        print ""
    print "/======================================="
    raw_input()

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

class Robot:
            
    BOARD_WIDTH = 19
    BOARD_HEIGHT = 19

    def compute_groups(self, game, diagonal=True):
        bots = game['robots']
        grouped_bots = {}
        ally_groups = []
        enemy_groups = []
        for loc, bot in bots.items():
            if bot.player_id == self.player_id:
                # if this bot has not yet been assigned to a group
                if not loc in grouped_bots:
                    # compute its group
                    mygroup = dfs_group(bots, loc, bot.player_id, [bot.location], diagonal)
                    # append to list of all groups
                    ally_groups.append(mygroup)
                    # mark all bots in this group as having-a-group
                    for group_loc in mygroup:
                        grouped_bots[group_loc] = mygroup
            else:
                if not loc in grouped_bots:
                    mygroup = dfs_group(bots, loc, bot.player_id, [bot.location], diagonal)
                    enemy_groups.append(mygroup)
                    for group_loc in mygroup:
                        grouped_bots[group_loc] = mygroup
        return ally_groups, enemy_groups

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
    
    def neighbors(self, game):
        allies = []
        enemies = []
        for near in rg.locs_around(self.location, filter_out=('invalid', 'obstacle')):
            if near in game['robots']:
                bot = game['robots'][near]
                if bot.player_id == self.player_id:
                    allies.append(bot)
                else:
                    enemies.append(bot)
        return allies, enemies

    def act(self, game):
        # first run: make the 'updated' attribute
        if not hasattr(self, 'groups_updated'):
            self.groups_updated = -1
            self.ally_groups = []
            self.enemy_groups = []
            self.taget_group = None
            self.target_location = rg.CENTER_POINT
            self.bfs_grid = []
        # 'self' is an entity shared by all bots. each frame, only one bot needs to make decisions, then they are shared
        if self.groups_updated < game['turn']:
            self.groups_updated = game['turn']
            self.ally_groups, self.enemy_groups = self.compute_groups(game, True)
            # sort groups by strength (sum of health)
            self.enemy_groups = sorted(self.enemy_groups, key=(lambda grp: sum([game['robots'][loc].hp for loc in grp])))
            # attack weakest group
            # TODO multiple targets
            self.target_group = self.enemy_groups[0]
            target_location_x = sum([loc[0] for loc in self.target_group])
            target_location_y = sum([loc[1] for loc in self.target_group])
            self.target_location = (target_location_x, target_location_y)
            self.bfs_grid = bfs_tree(game['robots'], self.target_location, self.player_id)
        # bot-specifics: move towards target or attack if there
        next = self.bfs_grid[self.location[0]][self.location[1]]
        # print self.location, "to", self.target_location, "via", next
        if next is not None:
            if next in game['robots']:
                return self.guard_or_attack(game)
            else:
                return ['move', next]
        elif rg.toward(self.location, self.target_location) in game['robots']:
            return ['attack', rg.toward(self.location, self.target_location)]
        else:
            return self.guard_or_attack(game)
