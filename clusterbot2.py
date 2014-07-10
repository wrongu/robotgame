import rg
import random, math
""" Available commands are
    ['move', (x, y)]
    ['attack', (x, y)]
    ['guard']
    ['suicide']
"""

""" game state looks like this
{
    'robots': {
        (x1, y1): {   
            'location': (x1, y1),
            'hp': hp,
            'player_id': player_id,
        },
        # ... other robots
    },

    # number of turns passed (starts at 0)
    'turn': turn
}
"""

def wander(location):
    direction = random.choice([0, math.pi/2, math.pi, -math.pi/2])
    return (round(location[0] + math.cos(direction)), round(location[1] + math.sin(direction)))

def request_move(bot, game, allies):
    # find some nearby allies
    nearby = [ally for ally in allies if rg.wdist(bot.location, ally.location) < Robot.NEARNESS_THRESHOLD]
    if len(nearby) > Robot.GROUP_MIN:
        # find center of nearby group (note that len(nearby)>=1 since at least 'self' is there)
        center_x = sum([n.location[0] for n in nearby]) / len(nearby)
        center_y = sum([n.location[1] for n in nearby]) / len(nearby)
        center = (center_x, center_y)
        # move towards center, but prioritize having weaker bots in the center
        next = rg.toward(bot.location, center)
        return next
    else:
        return wander(bot.location)

class Robot:
    
    NEARNESS_THRESHOLD = 15
    GROUP_MIN = 2
    
    def act(self, game):
        # get a list of all my allies
        allies = [bot for bot in game['robots'].values() if bot.player_id == self.player_id]
        # process movement alone
        movements = {}
        attacks = {}
        
        for bot in allies:
            movements[bot.location] = request_move(bot, game, allies)
        # remove collisions
        for loc, choice in movements.items():
            # check for environment collisions
            if 'obstacle' in rg.loc_types(choice) or 'invalid' in rg.loc_types(choice):
                del movements[loc]
                continue
            occupant = game['robots'].get(choice)
            # check for vacancy
            if occupant != None:
                del movements[loc]
                # check for enemy
                if occupant.player_id != self.player_id:
                    attacks[loc] = occupant.location
        # enact MY decision
        if self.location in movements:
            return ['move', movements[self.location]]
        elif self.location in attacks:
            return ['attack', attacks[self.location]]
        # neither move nor attack. if I will die in 1 turn and there are nearby enemies, suicide
        elif self.hp <= 18 and True in [loc in game['robots'] and game['robots'][loc].player_id != self.player_id for loc in rg.locs_around(self.location)]:
            return ['suicide']
        else:
            return ['guard']
