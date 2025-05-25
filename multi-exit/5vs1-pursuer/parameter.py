REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 400  # the number of nodes will be padded to this value
K_SIZE = 9  # the number of neighboring nodes

USE_GPU = True  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 2
NUM_META_AGENT = 32
LR = 1e-5
DECAY_STEP = 256
GAMMA = 0.99
SUMMARY_WINDOW = 1
LOAD_MODEL = False # do you want to load the model trained before
SAVE_IMG_GAP = 101
N_ROBOTS = 5
train_mode = False
FIXED_OPPONENT = False
BETA = 0.1
RANDOM_SEED = None
INPUT_TYPE = 'adj'
EXIT_NUM = 8 
INPUT_DIM = N_ROBOTS + EXIT_NUM + 1
if EXIT_NUM > 0:
    MIN_EVADOR_EXIT_DIST = 6

extra_info = ''
FOLDER_NAME = '5V1-EXIT{}-BETA{:.2f}-{}'.format(
    EXIT_NUM,
    BETA,
    extra_info
)
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
