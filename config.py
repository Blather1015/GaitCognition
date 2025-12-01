# config.py

DATASET_ROOT = r"C:\Work\GaitCognition\SampleDataset"  
OUTPUT_DATASET = r"./dataset/processed"

NUM_JOINTS = 24
INPUT_FEATURES = 3     # X, Y, Z
SEQ_LEN = 75           # ใช้ 75 เฟรมต่อคลิป
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0005
