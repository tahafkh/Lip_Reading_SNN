from SNN_models import *
from utils import *
import numpy as np
import torch
import os
from spikingjelly.activation_based import functional, surrogate, neuron
import tonic
from spikingjelly.activation_based import layer
import random
import argparse
from torch.utils.data import DataLoader
import time

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
parser.add_argument("-d", dest="is_delayed", action="store_true", default=False, help="delayed network")
parser.add_argument("--axonal", dest="has_axonal_delay", action="store_true", default=False, help="axonal-delayed network")
parser.add_argument("--dendritic", dest="has_dendritic_delay", action="store_true", default=False, help="dendritic-delayed network")
parser.add_argument("--analysis", action="store_true", default=False, help="analysis existing model")
parser.add_argument("--round", action="store_true", default=False, help="round positions")
parser.add_argument('--checkpoint_name', type=str, help="checkpoint model name")
parser.add_argument("--change", action="store_true", default=False, help="change state dict")

# dataset
parser.add_argument('--dataset', type=str, required=False, default="dvs_lip")
parser.add_argument('--dataset_path', type=str, required=False, default="/home/hugo/Work/TER/DVS-Lip")
parser.add_argument('--n_class', type=int, default=100)

# model
parser.add_argument('--model_name', type=str, default="spiking_mstp_low")

args = parser.parse_args()

LR = 1e-3 if not args.lr else args.lr
BATCH_SIZE = 32 if not args.batch_size else args.batch_size
MODEL_CHECKPOINT_PATH = "checkpoint.pt"
BEST_MODEL_CHECKPOINT_PATH = "best.pt"
NUM_CLASSES = 100 if not args.n_class else args.n_class
EPOCHS  = 100 if not args.max_epoch else args.max_epoch
RESUME_TRAINING = args.resume_training # If true, will load the model saved in MODEL_CHECKPOINT_PATH 
DATASET_PATH="/home/hugo/Work/TER/DVS-Lip" if not args.dataset_path else args.dataset_path
T = args.T
#DATASET_PATH="/home/hugo/Work/TER/i3s_dataset3"

MODEL_BASE_PATH = os.path.expanduser('~/paper_runs')

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
	model = LowRateBranch(n_class=NUM_CLASSES, spiking_neuron=plif, delayed=args.is_delayed, axonal_delay=args.has_axonal_delay, dendritic_delay=args.has_dendritic_delay, detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
elif args.model_name=="snn1":
	model = SNN1(n_class=NUM_CLASSES, spiking_neuron=plif,  detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
elif args.model_name=="snn2":
	model = SNN2(n_class=NUM_CLASSES, spiking_neuron=plif,  detach_reset=True, surrogate_function=surrogate.Erf(), step_mode='m').to(DEVICE)
else:
	print("--model_name should be either spiking_mstp_low, snn1, or snn2")
	exit()

functional.set_step_mode(model, 'm')

position_params = []
others_params = []
for name, param in model.named_parameters():
	if name.endswith(".P") or name.endswith(".SIG") and param.requires_grad:
		position_params.append(param)
	else:
		others_params.append(param)
param_groups = [
	{"params": others_params, "lr": LR, "weight_decay": 10e-7},
	{"params": position_params, "lr": LR * 100, "weight_decay": 0.0},
]
optimizer = torch.optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# Set scheduler to None if you don't want to use it
#scheduler = None
start_epoch = 0

# Loading MODEL_CHECKPOINT_PATH if resume training is true
if RESUME_TRAINING:
	checkpoint= torch.load(MODEL_CHECKPOINT_PATH)
	model.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	start_epoch = checkpoint['epoch']
	print("Resuming training from epoch", start_epoch)
	if scheduler is not None:
		scheduler.load_state_dict(checkpoint['scheduler'])

# Print the memory taken by the model
model_memory_need = model_memory_usage(model)
print("Model memory usage: ", model_memory_need, "bytes", "->", model_memory_need*0.000001, "MB")

training_losses = []
mean_losses = []
test_losses = []
accuracies = []
best_epoch = {"accuracy":0, "val_loss":9999, "train_loss":9999, "epoch":0}

torch.autograd.set_detect_anomaly(True)

if not args.analysis:
	# Training/testing loop
	for epoch in trange(start_epoch, EPOCHS):
		epoch_start = time.time()
		train_loss, train_accuracy = train(model, DEVICE, train_loader, optimizer, num_labels=NUM_CLASSES, scheduler=scheduler)
		model.decrease_sig(epoch, EPOCHS)
		test_loss, accuracy = test(model, DEVICE, test_loader, num_labels=NUM_CLASSES)

		training_losses.append(train_loss)
		mean_losses.append(train_loss)
		test_losses.append(test_loss)
		accuracies.append(accuracy)
		checkpoint={
			'epoch':epoch,
			'model':model.state_dict(),
			'optimizer':optimizer.state_dict(),
			'scheduler': None if scheduler is None else scheduler.state_dict()
		}
		torch.save(checkpoint, MODEL_CHECKPOINT_PATH)

		# We save the best model in BEST_MODEL_CHECKPOINT_PATH
		if accuracy>best_epoch["accuracy"] or (accuracy==best_epoch["accuracy"] and test_loss<best_epoch["val_loss"]):
			best_epoch["accuracy"]=accuracy
			best_epoch["val_loss"]=test_loss
			best_epoch["train_loss"]=train_loss
			best_epoch["epoch"]=epoch
			checkpoint = {
				'epoch':epoch,
				'model':model.state_dict(),
				'optimizer':optimizer.state_dict(),
				'scheduler': None if scheduler is None else scheduler.state_dict()
			}
			torch.save(checkpoint, BEST_MODEL_CHECKPOINT_PATH)

		epoch_end = time.time()
		epoch_duration = epoch_end - epoch_start
		print("Epoch (training) took {:.3f}s".format(epoch_duration))

		print("Train loss at epoch", epoch, ":", train_loss)
		print("Train accuracy at epoch", epoch, ":", train_accuracy, "%")
		print("Test loss at epoch", epoch, ":", test_loss)
		print("Test accuracy at epoch", epoch, ":", accuracy, "%")
		print("BEST EPOCH SO FAR:",best_epoch)

		with open('res_test.txt', 'a') as f:
			f.write(
				"epoch %i: train: %.2f, val: %.2f, train_loss: %.5f, test_loss: %.3f, lr: %.5f, epoch duration: %.3f\n"
				% (
					epoch+1,
					train_accuracy, 
					accuracy, 
					train_loss, 
					test_loss, 
					optimizer.param_groups[0]["lr"],
					epoch_duration
				)
			)

	print("Training done !")
	print("BEST EPOCH:",best_epoch)
	print("Best model saved in", BEST_MODEL_CHECKPOINT_PATH)
	with open('res_test.txt', 'a') as f:
		f.write("best epoch, accu (val): %i %.2f"%(best_epoch["epoch"] +1, best_epoch["accuracy"]))
		f.write('\n')

else:
	model_path = os.path.join(MODEL_BASE_PATH, args.checkpoint_name + '.pt')
	model_state_dict = torch.load(model_path, map_location=DEVICE)
	if 'model' in model_state_dict.keys():
		model_state_dict = model_state_dict['model']
	if args.change:
		keys = list(model_state_dict.keys())
		for key in keys:
			new_key = key
			if 'downsample.0' in key:
				new_key = key.replace('downsample.0.', 'downsample.0.conv.')
			elif 'downsample_block' in key:
				new_key = key.replace('downsample_block.', 'downsample_block.conv.')
			elif 'conv' in key and 'layer' in key:
				phrases = key.rsplit('.', 1)
				new_key = phrases[0] + '.conv.' + phrases[1]
			model_state_dict[new_key] = model_state_dict.pop(key)
	
	model.load_state_dict(model_state_dict, strict=True)
	if args.round:
		with torch.no_grad():
			model.round_pos()
	print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
	test_loss, accuracy = test(model, DEVICE, test_loader, num_labels=NUM_CLASSES)
	print(f'Test Accuracy: {accuracy}')
	print('###############################')
