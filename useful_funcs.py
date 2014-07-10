
BOARD_WIDTH = 19
BOARD_HEIGHT = 19
HALF_RANGE = 2

def compute_territory_value(self, game):
    # initialize 2D grid of zeros
    territory = [None] * Robot.BOARD_HEIGHT
    for i in range(Robot.BOARD_HEIGHT):
        territory[i] = [0] * BOARD_WIDTH
    # each bot's territory score is equal to its health and diffuses out to nearby squares
    for loc, bot in game['robots'].items():
        sign = bot.player_id == self.player_id ? 1 : -1
        r = loc[0] - 1
        c = loc[1] - 1
        for i in range(r - Robot.HALF_RANGE, r + Robot.HALF_RANGE+1):
            for j in range(c - Robot.HALF_RANGE, c + Robot.HALF_RANGE+1):
                if 'normal' in rg.loc_types((i,j)):
                    territory[i][j] += sign * bot.hp / pow(2, rg.wdist(bot.location, (i,j)))
    return territory
    
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
                mygroup = Robot.dfs_group(bots, loc, bot.player_id, [bot], diagonal)
                # append to list of all groups
                ally_groups.append(mygroup)
                # mark all bots in this group as having-a-group
                for group_loc in mygroup:
                    grouped_bots[group_loc] = mygroup
        else:
            if not loc in grouped_bots:
                mygroup = Robot.dfs_group(bots, loc, bot.player_id, [bot], diagonal)
                enemy_groups.append(mygroup)
                for group_loc in mygroup:
                    grouped_bots[group_loc] = mygroup
    return ally_groups, enemy_groups
                  
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
        for pt in rg.locs_around(loc, filter_out=('invalid', 'obstacle'))
            if 'normal' in rg.loc_types(pt) and pt in bots and bots[pt].player_id == player and pt not in partial_group:
                # add it to the group and recurse
                partial_group.append((i,j))
                partial_group = dfs_group(bots, (i,j), player, partial_group, diagonal)
    return partial_group

def smart_toward(game, start, end, my_id):
    # initialize 2D grid of zeros
    grid = [None] * Robot.BOARD_HEIGHT
    for i in range(Robot.BOARD_HEIGHT):
        grid[i] = [None] * BOARD_WIDTH
    Q = Queue.Queue()
    Q.put(end)
    grid = bfs_tree(game['robots'], grid, Q, my_id)
    # check if path existed:
    if grid[end[0]][end[1]] is not None:
        fro = end, to = end
        while fro != start:
            to = fro
            fro = grid[fro[0]][fro[1]]
        return to
    else:
        return None
    
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