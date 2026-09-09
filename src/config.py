# Paths
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/valid"
TEST_DIR = "../data/test"
MODEL_SAVE_PATH = "../models/best_model_resnet.pth"

TRAIN_CSV = "../dataRaw/fer2013new.csv"
VAL_CSV = "../dataRaw/fer2013new.csv"      # isti fajl, filtrira se po Usage koloni
TEST_CSV = "../dataRaw/fer2013new.csv"

TRAIN_IMAGE_DIR = "../dataRaw/FER2013Train"
VAL_IMAGE_DIR = "../dataRaw/FER2013Valid"
TEST_IMAGE_DIR = "../dataRaw/FER2013Test"

NUM_CLASSES = 6 
# Dataset
NUM_CLASSES = 6
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = "cuda"

# Model
PRETRAINED = True
FREEZE_BACKBONE = True
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"

# Reproducibility
SEED = 42