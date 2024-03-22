from SNN_models import *
from utils import *
import os
import datetime
import numpy as np
import json
import torch
from spikingjelly.activation_based import functional, surrogate, neuron
import tonic
from spikingjelly.activation_based import layer
import random
import argparse
from torch.utils.data import DataLoader

# Setting some seeds for reproductibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, required=False, default=1e-3)
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('-T', type=int, default=30)
parser.add_argument('--max_epoch', type=int, required=False, default=100)
parser.add_argument('--resume_training',action='store_true')

# dataset
parser.add_argument('--dataset', type=str, required=False, default="dvs_lip")
parser.add_argument('--dataset_path', type=str, required=False, default="/home/hugo/Work/TER/DVS-Lip")
parser.add_argument('--n_class', type=int, default=100)

# model
parser.add_argument('--model_name', type=str, default="spiking_mstp_low")

args = parser.parse_args()

LR = 1e-3 if not args.lr else args.lr
BATCH_SIZE = 4 if not args.batch_size else args.batch_size

TIME = datetime.datetime.now().isoformat()
# TIME = '2024-03-19T01:49:45.647774'
BASE_PATH = os.path.expanduser(f'~/dvs-runs/{TIME}')
os.makedirs(BASE_PATH, exist_ok=True)

MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, 'full_3_acc_last_model.pth')
BEST_MODEL_CHECKPOINT_PATH = os.path.join(BASE_PATH, 'full_3_acc_best_model.pth')
RESULTS_PATH = os.path.join(BASE_PATH, 'full_3_acc.json')

NUM_CLASSES = 100 if not args.n_class else args.n_class
EPOCHS  = 100 if not args.max_epoch else args.max_epoch
RESUME_TRAINING = False # If true, will load the model saved in MODEL_CHECKPOINT_PATH 
DATASET_PATH="/home/hugo/Work/TER/DVS-Lip" if not args.dataset_path else args.dataset_path
T = args.T
#DATASET_PATH="/home/hugo/Work/TER/i3s_dataset3"

# We can either use the DVS-Lip or I3S dataset
if args.dataset=="dvs_lip":
	X_train = DVSLip_Dataset(dataset_path=DATASET_PATH, transform=center_random_crop, train=True, T=T)
	X_test = DVSLip_Dataset(dataset_path=DATASET_PATH, transform=center_crop, train=False, T=T)
elif args.dataset=="i3s":
	X_train = i3s_Dataset(dataset_path=DATASET_PATH, transform=center_random_crop, train=True, T=T)
	X_test = i3s_Dataset(dataset_path=DATASET_PATH, transform=center_crop, train=False, T=T)
else:
	print("--dataset should be either dvs_lip or i3s")
	exit()

train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(X_test, batch_size=BATCH_SIZE, shuffle=False)



if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
	print("Found GPU")
else:
	DEVICE = torch.device("cpu")

print(torch.cuda.get_device_name(DEVICE))
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_reserved(0))
print(torch.cuda.memory_allocated(0))
print(torch.cuda.mem_get_info(0))

lif = neuron.LIFNode
plif = neuron.ParametricLIFNode

# Define the model to use
if args.model_name=="spiking_mstp_low":
	model = LowRateBranch(n_class=NUM_CLASSES, spiking_neuron=plif,  detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
elif args.model_name=="snn1":
	model = SNN1(n_class=NUM_CLASSES, spiking_neuron=plif,  detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
elif args.model_name=="snn2":
	model = SNN2(n_class=NUM_CLASSES, spiking_neuron=plif,  detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
else:
	print("--model_name should be either spiking_mstp_low, snn1, or snn2")
	exit()

functional.set_step_mode(model, 'm')

position_params = []
other_params = []
for name, param in model.named_parameters():
    if name.endswith('.P') and param.requires_grad:
        position_params.append(param)
    else:
        other_params.append(param)

param_groups = [{'params': position_params,'lr': LR*10},
                {'params': other_params,'lr': LR},
               ]

optimizer = torch.optim.Adam(param_groups, lr=LR, weight_decay=10e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# Set scheduler to None if you don't want to use it
#scheduler = None
start_epoch = 0

# Print the memory taken by the model
model_memory_need = model_memory_usage(model)
print("Model memory usage: ", model_memory_need, "bytes", "->", model_memory_need*0.000001, "MB")

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
best_epoch = {"accuracy":0, "val_loss":9999, "train_loss":9999, "epoch":0}

torch.autograd.set_detect_anomaly(True)

# Loading MODEL_CHECKPOINT_PATH if resume training is true
if RESUME_TRAINING:
    checkpoint= torch.load(MODEL_CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    if scheduler is not None:
    	scheduler.load_state_dict(checkpoint['scheduler'])
    with open(RESULTS_PATH, 'r') as f:
        results = json.load(f)
    train_losses = results['train_losses']
    test_losses = results['test_losses']
    train_accuracies = results['train_accuracies']
    test_accuracies = results['test_accuracies']
    best_epoch = results['best_epoch']
    print("Resuming training from epoch", start_epoch)

# Training/testing loop
for epoch in trange(start_epoch, EPOCHS):
    train_loss, train_accuracy = train(model, DEVICE, train_loader, optimizer, num_labels=NUM_CLASSES, scheduler=scheduler)
    model.decrease_sig(epoch, EPOCHS)
    test_loss, test_accuracy = test(model, DEVICE, test_loader, num_labels=NUM_CLASSES)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    checkpoint = {
        'epoch':epoch,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict()
    }
    torch.save(checkpoint, MODEL_CHECKPOINT_PATH)

	# We save the best model in BEST_MODEL_CHECKPOINT_PATH
    if test_accuracy>best_epoch["accuracy"] or (test_accuracy==best_epoch["accuracy"] and test_loss<best_epoch["val_loss"]):
        best_epoch["accuracy"]=test_accuracy
        best_epoch["val_loss"]=test_loss
        best_epoch["train_loss"]=train_loss
        best_epoch["epoch"]=epoch
        torch.save(checkpoint, BEST_MODEL_CHECKPOINT_PATH)
    
    print("Train loss at epoch", epoch, ":", train_loss)
    print("Train accuracy at epoch", epoch, ":", train_accuracy, "%")
    print("Test loss at epoch", epoch, ":", test_loss)
    print("Test accuracy at epoch", epoch, ":", test_accuracy, "%")
    print("BEST EPOCH SO FAR:",best_epoch)

    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_epoch': best_epoch,
    }

    with open(RESULTS_PATH, 'w') as file:
        json.dump(results, file)

print("Training done !")
print("BEST EPOCH:",best_epoch)
