import os
import torch
import torch.multiprocessing
import torchvision.transforms as transforms

from alexnet import model
from utils import training_loop


if __name__ == '__main__':

    MODEL_NAME = 'alexnet'
    DATA_ROOT = 'data'
    DATASET_NAME = 'imagenette2_sample'
    INPUT_ROOT_DIR = os.path.join(DATA_ROOT, DATASET_NAME)
    OUTPUT_DIR = os.path.join(DATA_ROOT, f'{MODEL_NAME}_data_out')
    
    TRAIN_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'train')
    VAL_IMG_DIR = os.path.join(INPUT_ROOT_DIR, 'val')
    
    LOG_DIR = os.path.join(OUTPUT_DIR, 'tblogs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR , 'models')
    
    NUM_EPOCHS = 45  
    BATCH_SIZE = 32
    MOMENTUM = 0.9
    LR_DECAY = 0.0005
    LR_INIT = 0.00001
    N_SAMPLE = 100
    N_CHECK = 5
    IMAGE_DIM = 227
    NUM_CLASSES = 2
    DEVICE_IDS = [0, 1, 2, 3] 
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    
    OPTIMIZER_NAME = 'Adam'
    OTHER_PARAMS = {'lr': LR_INIT, }

    TRANSFORMS_LIST = [transforms.CenterCrop(IMAGE_DIM),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=MEANS, std=STDS),]

    training_loop(LOG_DIR, NUM_CLASSES, MODEL_NAME, 
                  DEVICE_IDS, TRAIN_IMG_DIR, VAL_IMG_DIR, TRANSFORMS_LIST,
                  BATCH_SIZE, N_SAMPLE, OPTIMIZER_NAME, OTHER_PARAMS, NUM_EPOCHS, N_CHECK, CHECKPOINT_DIR)