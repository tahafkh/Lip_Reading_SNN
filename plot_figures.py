import torch
import matplotlib.pyplot as plt
import argparse
import os

MODEL_BASE_PATH = os.path.expanduser('~/paper_runs')
FIGURE_BASE_PATH = os.path.expanduser('~/figures')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)

args = parser.parse_args()

model_figure_dir = os.path.join(FIGURE_BASE_PATH, args.model_name)
os.mkdir(model_figure_dir)

model_path = os.path.join(MODEL_BASE_PATH, args.model_name + '.pt')
model = torch.load(model_path, map_location='cpu')
if 'model' in model.keys():
    model = model['model']
for param in model.keys():
    if param.endswith('.P'):
        positions = model[param].flatten()

        plt.figure()
        plt.hist(positions.numpy())
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        figure_name = param.replace('.', '_') + '.png'
        plt.savefig(os.path.join(model_figure_dir, figure_name))
        