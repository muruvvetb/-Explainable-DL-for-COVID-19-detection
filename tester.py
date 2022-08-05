import torch
import os
import numpy as np
import argparse
import json
from utils import set_seed
from dataprep import get_dataloader
from model import DecoderEncoderDecoder
from tqdm import tqdm
from main import eval

def tester(args, device):
    
    batch_size = args.batch_size
    _, _, test_loader = get_dataloader(batch_size, args.dataset_path, args.csv_path, image_size=224, device=device)
    num_classes = len(set(test_loader.dataset.target_classes))
    model = DecoderEncoderDecoder(num_classes, args.arch, hidden_size=64, dropout_rate=0.5, pretrained=True)
    model_names = ["best_model.pt", "final_model.pt"]
    results = {}
    for model_name in model_names:
        model.load_state_dict(torch.load(f"{args.path}/{model_name}" , map_location=device))
        model.to(device)
        model.eval()
        test_acc, test_acc_expl = eval(model, test_loader, device)

        results[model_name] = {"test_acc": test_acc, "test_acc_expl": test_acc_expl}

    with open(f"{args.path}/test_results.txt", "w") as f:
        json.dump(results, f, default=str)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="DeiTSmall",
                        choices=["DeiTBase", "DeiTTiny", "DeiTSmall", "ResNet50", "ResNet18", "ResNet101", "ResNet152"])
    parser.add_argument("--dataset_path", default="./dataset/")
    parser.add_argument("--csv_path", default="./csv_files/")
    parser.add_argument("--path", default="results/2021 12 23 - 15 42 01")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1773)
    parser.add_argument("--save", type=int, default=1)

    args = parser.parse_args()
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tester(args, device)
