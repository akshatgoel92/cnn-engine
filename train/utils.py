import os
import torch
import random
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms


import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from train.alexnet.model import AlexNet



def set_seed():
    """
    Set random seed
    """
    return torch.initial_seed()


def get_model(NUM_CLASSES, device, model_name):
    """
    Setup model
    """
    if model_name == 'alexnet':
        return AlexNet(num_classes=NUM_CLASSES).to(device)


def set_data_parallel(model, DEVICE_IDS):
    """
    Setup data parallel
    settings
    """
    return torch.nn.parallel.DataParallel(model, device_ids=DEVICE_IDS)


def get_dataset(IMG_DIR, transforms_list):
    """
    Get dataset
    """
    return datasets.ImageFolder(IMG_DIR, transforms.Compose(transforms_list))


def get_dataloader(dataset, BATCH_SIZE, shuffle=True, 
                   pin_memory=True, num_workers=4, 
                   drop_last=True, subset_data=False, 
                   N_SAMPLE=None):
    """
    Get data-loader
    """
    if subset_data == True:
        assert N_SAMPLE is not None
        subset = random.sample(range(len(dataset)), N_SAMPLE)
        dataset = data.Subset(dataset, subset)

    return data.DataLoader(
                    dataset,
                    shuffle=shuffle,
                    pin_memory=pin_memory,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    batch_size=BATCH_SIZE)


def get_optimizer(optimizer_name, model, other_params):
    """
    Create optimizer
    """
    params = model.parameters()
    if optimizer_name == 'SGD':
        return optim.SGD(params, **other_params)
    if optimizer_name == 'Adam':
        return optim.Adam(params=model.parameters(), **other_params)


def set_lr_schedule(optimizer, step_size=1, gamma=0.1, policy_name='Step', step_size_up=10, base_lr=0.01, max_lr=0.4):
    """
    Set LR scheduler
    """
    if policy_name == 'Step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif policy_name == 'Cyclic':
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up)
        



def save_checkpoint(model_name, CHECKPOINT_DIR, epoch, 
                   total_steps, optimizer, model, seed, history):
    """
    Save checkpoint
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best_loss.pkl')
    state = {
                #'epoch': epoch,
                #'total_steps': total_steps,
                #'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
                #'seed': seed,
                #'history': history,
            }
    torch.save(state, checkpoint_path)


def training_loop(LOG_DIR, NUM_CLASSES, model_name, 
                  DEVICE_IDS, TRAIN_IMG_DIR, VAL_IMG_DIR, transforms_list,
                  BATCH_SIZE, N_SAMPLE, optimizer_name, other_params, NUM_EPOCHS,
                  N_CHECK, CHECKPOINT_DIR):
    """
    This is the main
    training and evaluation
    loop for the model
    """
    # Misc. settings to make sure no errors happen
    torch.autograd.set_detect_anomaly(True)

    # Prevents out of memory error when num_workers > 1 in data loaders
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Set devices to be agnostic to whether CPU or GPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Make directory to store checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Print the seed value
    seed = set_seed()
    print('Used seed : {}'.format(seed))

    # For Tensorboard
    tbwriter = set_tensorboard(LOG_DIR)
    print('TensorboardX summary writer created')

    # Create model
    model = get_model(NUM_CLASSES, device, model_name)
    print('Model instance created')

    # Train on multiple GPUs
    model = set_data_parallel(model, DEVICE_IDS)
    print('Model data parallelism set up successfully..')

    
    # Create datasets and data loaders
    train_dataset = get_dataset(TRAIN_IMG_DIR, transforms_list)
    print('Train dataset created')
    
    
    val_dataset = get_dataset(VAL_IMG_DIR, transforms_list)
    print('Val dataset created')
    

    train_loader = get_dataloader(train_dataset, BATCH_SIZE, 
                                  shuffle=True, pin_memory=True, 
                                  num_workers=4, drop_last=True, 
                                  subset_data=False, N_SAMPLE=None)




    train_evaluator = get_dataloader(train_dataset, len(train_dataset), 
                                     shuffle=True, pin_memory=True, 
                                     num_workers=4, drop_last=True, 
                                     subset_data=False, N_SAMPLE=None)

    for i, data in enumerate(train_evaluator):
        X_train, Y_train = data


    val_loader = get_dataloader(val_dataset, len(val_dataset), 
                                shuffle=True, pin_memory=True, 
                                num_workers=4, drop_last=True, 
                                subset_data=False, N_SAMPLE=None)



    for i, data in enumerate(val_loader):
        X_val, Y_val = data
    

    # Print 
    print('Dataloaders created')

    optimizer = get_optimizer(optimizer_name, model, other_params)
    print('Optimizer created')


    # Multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = set_lr_schedule(optimizer, step_size=30, gamma=0.1)
    print('LR Scheduler created...')

    # Start training!
    total_steps = 1
    print('Starting training...')

    # Set whether to run in test mode
    test = False

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize the dictionary where all metrics will be stored
    history = {
        'train_accuracy': [],
        'train_loss': [], 
        'val_accuracy': [],
        'val_loss': []
    }

    # Training loop
    if not test:

        min_val_loss = 999999999
        # Train for maximum number of epochs given here
        for epoch in tqdm(range(NUM_EPOCHS)):
            # Initialize running loss value for this epoch
            running_loss = 0.0
            # Loop through the generator containing training images
            for i, data in enumerate(train_loader):
              # Retrieve inputs and labels
              inputs, labels = data
              # Send them to GPU 
              inputs = inputs.type(torch.FloatTensor).to(device)
              # Labels
              labels =  labels.type(torch.LongTensor).to(device)
              # Zero the optimizer gradients
              optimizer.zero_grad()
              # Forward propagation
              outputs = model.forward(inputs)
              # Compute loss
              loss = loss_fn(outputs, labels)
              # Backward propogation
              loss.backward()
              # Weight update
              optimizer.step()
      
            # Compute training loss
            with torch.no_grad():
              # Forward propagation
              out = model.forward(X_train)
              # Prediction 
              preds = out.argmax(axis=1)
              # Get accuracy 
              accuracy = sum(preds == Y_train)/len(Y_train)
              # Get loss 
              loss = loss_fn(out, Y_train)
              # Print
              print("Train accuracy {}, Train Loss: {}".format(accuracy, loss))
              # Append to history record
              history['train_accuracy'].append(accuracy)
              # Append loss to history record
              history['train_loss'].append(loss)

          
            # Compute validation loss
            with torch.no_grad():
              # Forward propogation
              out = model.forward(X_val)
              # Prediction 
              preds = out.argmax(axis=1)
              # Calculate accuracy
              accuracy = sum(preds == Y_val)/len(Y_val)
              # Calculate loss 
              loss = loss_fn(out, Y_val)
              # Print statement
              print("Val. accuracy {}, Val. Loss:, {}".format(accuracy, loss))
              # Append accuracy to history record
              history['val_accuracy'].append(accuracy)
              # Append loss
              history['val_loss'].append(loss)

            # If loss has improved
            if loss < min_val_loss:
              # Save checkpoint
              save_checkpoint(model_name, CHECKPOINT_DIR, epoch, 
                              total_steps, optimizer, model, seed, history)
              # Minimum validation loss
              min_val_loss = loss


def testing_loop(IMG_PATH, NUM_CLASSES, MODEL_PATH, transforms_list, DEVICE_IDS=[0, 1, 2, 3, 4]):
  """
  Test on single image
  """
  img = Image.open(IMG_PATH)
  model = AlexNet(NUM_CLASSES)
  model = set_data_parallel(model, DEVICE_IDS)
  model.load_state_dict(torch.load(MODEL_PATH)['model'])
  
  transform = transforms.Compose(transforms_list)
  img_tensor = transform(img).unsqueeze(0)
  
  
  model.eval()

  with torch.no_grad():
    out = model.forward(img_tensor)
    prediction = out.argmax(axis=1)

  return prediction