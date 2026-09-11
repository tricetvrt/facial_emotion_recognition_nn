# Paths
TRAIN_DIR_HARD = "../data/train"
VAL_DIR_HARD = "../data/valid"
TEST_DIR_HARD = "../data/test"
MODEL_SAVE_PATH = "../models/best_model_resnet.pth"

TRAIN_CSV = "../dataRaw/fer2013new.csv"
VAL_CSV = "../dataRaw/fer2013new.csv"      # isti fajl, filtrira se po Usage koloni
TEST_CSV = "../dataRaw/fer2013new.csv"

TRAIN_IMAGE_DIR = "../dataClean/FER2013Train"
VAL_IMAGE_DIR = "../dataClean/FER2013Valid"
TEST_IMAGE_DIR = "../dataClean/FER2013Test"

TRAIN_AUG_DIR = "../dataAugmented/FER2013Train"



# Dataset
NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
USE_SOFT_LABELS= False
USE_WEIGHTED_SAMPLER = False

# Training
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = "cuda"
USE_AUGMENTATION = False

# Model
PRETRAINED = True
FREEZE_BACKBONE = True
MODEL_NAME = "resnet50"
# "mobilenetv4_conv_small.e2400_r224_in1k"

# Reproducibility
SEED = 42