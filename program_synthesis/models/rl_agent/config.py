from program_synthesis.datasets.karel.mutation import ACTION_NAMES, BLOCK_TYPE, CONDS, REPEAT_COUNTS

DISCOUNT = 0.99
EPSILON = 0.1
ALPHA = 0.7

VOCAB_SIZE = 43
MAX_TOKEN_PER_CODE = 20

# Number of different possible actions to modify the code
TOTAL_MUTATION_ACTIONS = 8

# Dimension of task latent space
# Task <=> (I/O)
TASK_EMBED_SIZE = 512

CODE_EMBED_SIZE = 512

# Dimension of state latent space
# State <=> (Task, Code Embed)
STATE_EMBED_SIZE = TASK_EMBED_SIZE + CODE_EMBED_SIZE

# Dimension of each location latent space
LOCATION_EMBED_SIZE = 32

# Token embed size
TOKEN_EMBED_SIZE = 256

KAREL_STATIC_TOKEN = len(ACTION_NAMES)
BLOCK_TYPE_SIZE = len(BLOCK_TYPE)
CONDITION_SIZE = len(CONDS)
REPEAT_COUNT_SIZE = len(REPEAT_COUNTS)
