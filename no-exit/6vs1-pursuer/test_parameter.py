EMBEDDING_DIM = 128
K_SIZE = 9  # the number of neighbors

USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 2  # the number of GPUs
NUM_META_AGENT = 20  # the number of processes

TEST_MAP = 'Grid'
NUM_TEST = 500
NUM_RUN = 1
SAVE_GIFS = 0
SAVE_TRAJECTORY = 0
SAVE_LENGTH = 0
N_ROBOTS = 6

FIXED_OPPONENT = False
RANDOM_SEED = None
INPUT_TYPE = 'adj'
POLICY_TYPE = 'RL'  # 'RL' or 'DP'
INPUT_DIM = N_ROBOTS + 1
extra_info = ''
FOLDER_NAME = '6V1-FO{:d}-BETA{:.2f}-{}'.format(
    FIXED_OPPONENT,
    0.1,
    extra_info
)
model_path = f'model/pursuer_model/'
gifs_path = f'results/{FOLDER_NAME}/gifs'
trajectory_path = f'results/{FOLDER_NAME}/trajectory/'
length_path = f'results/length'