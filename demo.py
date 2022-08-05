import torch
import os
import numpy as np
import argparse
import time
import json
from utils import set_seed, Logger
from dataprep import get_dataloader
from model import DecoderEncoderDecoder
from tqdm import tqdm
from main import eval

batch_size = 64
train_loader, val_loader, test_loader = get_dataloader(batch_size, "./dataset/", "./csv_files/", image_size=224)
num_classes = len(set(train_loader.dataset.target_classes))
model = DecoderEncoderDecoder(num_classes, "ResNet50", hidden_size=64, dropout_rate=0.5, pretrained=True)


"""os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'"""
device = 'cpu'
model.to(device)
model.load_state_dict(torch.load("results/2021 12 19 - 02 41 18/best_model.pt", map_location=device))
device = 'cpu'
model.to(device)
model.eval()

eval(model, test_loader, device)
