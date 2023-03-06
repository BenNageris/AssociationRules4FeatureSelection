import os.path

# relative datasets directory
DATASETS_DIRECTORY = r"datasets"
# path format to the training file
TRAIN_DATASET = os.path.join(DATASETS_DIRECTORY, f"{{}}", "train.csv")
# path format to the testing file
TEST_DATASET = os.path.join(DATASETS_DIRECTORY, f"{{}}", "test.csv")
