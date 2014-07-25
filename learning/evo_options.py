# instead of cmd line args, set evolution_bot configuration here

# tournament config
game_seed = 321654
num_games = 1

# populations config
pop0 = "Individuals"
pop0_size = 1
pop0_random = 0.05
pop0_mutation = 0.01
pop0_seed = None # file from which to load brains
pop0_saveto = "evolved_brains/sim1/red"
pop0_gen_length = 0 # if greater than 0, then every <this many> turns, next_generation() is called
pop0_slate_rand = 0.01

pop1 = "Family"
pop1_size = 1
pop1_random = 0.05
pop1_mutation = 0.01
pop1_seed = None # file from which to load brains
pop1_saveto = "evolved_brains/sim1/blue"
pop1_gen_length = 0 # if greater than 0, then every <this many> turns, next_generation() is called
pop1_slate_rand = 0.01

from rgkit.settings import settings
settings.spawn_per_player = 1
settings.spawn_every = 15
settings.max_turns = 1000
