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


def train(model, train_loader, epoch, opt, device):
    model.train()

    total_loss = 0.0
    total_item = 0
    for data, target in tqdm(train_loader, disable=False):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        result = model.loss(data, target, epoch)
        loss = result["loss"]
        loss.backward()
        total_loss += loss.item()
        total_item += data.shape[0]
        opt.step()

    return total_loss / total_item


def eval(model, loader, device):
    model.eval()
    correct = 0
    correct_expl = 0
    total_item = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            classes, classes_explanations, explanations = model.predict(data)
            correct += classes.eq(target.view_as(classes)).sum().item()
            correct_expl += classes_explanations.eq(target.view_as(classes_explanations)).sum().item()
            total_item += data.shape[0]

    return 100 * correct / total_item, 100 * correct_expl / total_item


def trainer(args, device, logger):
    logger(f'Train started at: {time.strftime("%Y %m %d - %H %M %S")}')
    start = time.time()
    batch_size = args.batch_size
    train_loader, val_loader, test_loader = get_dataloader(batch_size, args.dataset_path, args.csv_path, image_size=224, device=device)
    num_classes = len(set(train_loader.dataset.target_classes))
    model = DecoderEncoderDecoder(num_classes, args.arch, hidden_size=64, dropout_rate=0.5, pretrained=True, decoder_mode=args.decoder_mode)
    logger(f"model: {model}")
    lrate = 1e-5
    logger(f"lrate: {lrate}")
    opt = torch.optim.Adam(model.parameters(), lr=lrate)
    logger(f"opt: {opt}")
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.max_epochs // 10, gamma=0.2)
    logger(f"scheduler: {scheduler}")

    model.to(device)
    results = {
        "train": {"loss": {"idx": [], "loss": []}},
        "val": {"acc": {"idx": [], "acc": []}, "expl_acc": {"idx": [], "expl_acc": []}},
        "test": {"acc": {"idx": [], "acc": []}, "expl_acc": {"idx": [], "expl_acc": []}}
    }
    progress_bar = tqdm(range(args.max_epochs), position=0)
    best_val_acc = -1
    for epoch in progress_bar:
        train_result = train(model, train_loader, epoch, opt, device)
        results["train"]["loss"]["idx"].append(epoch)
        results["train"]["loss"]["loss"].append(train_result)

        val_acc, val_acc_expl = eval(model, val_loader, device)
        results["val"]["acc"]["idx"].append(epoch)
        results["val"]["acc"]["acc"].append(val_acc)

        results["val"]["expl_acc"]["idx"].append(epoch)
        results["val"]["expl_acc"]["expl_acc"].append(val_acc_expl)

        to_print = f'Epoch: {epoch}: || Train Loss: {train_result:.6f}, Val Acc: {val_acc:.3f}, Val Expl Acc: {val_acc_expl:.3f}'

        if epoch % 5 == 0 or epoch == args.max_epochs - 1:
            test_acc, test_acc_expl = eval(model, test_loader, device)
            results["test"]["acc"]["idx"].append(epoch)
            results["test"]["acc"]["acc"].append(test_acc)

            results["test"]["expl_acc"]["idx"].append(epoch)
            results["test"]["expl_acc"]["expl_acc"].append(test_acc_expl)
            to_print += f', Test Acc: {test_acc:.3f}, Test  Expl Acc: {test_acc_expl:.3f}'

            logger.plot(results)

        progress_bar.set_description(desc=to_print)
        logger(to_print)

        if best_val_acc < val_acc:
            torch.save(model.state_dict(), logger.path + "/best_model.pt")
            best_val_acc = val_acc

    torch.save(model.state_dict(), logger.path + "/final_model.pt")
    logger.write(results)

    logger(f'Train finished at: {time.strftime("%Y %m %d - %H %M %S")}')
    duration = time.time() - start
    logger(f'Train took {duration // 3600} hour {(duration % 3600) // 60} min {int(duration % 60)} sec')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="ResNet50",
                        choices=["DeiTBase", "DeiTTiny", "DeiTSmall", "ResNet50", "ResNet18", "ResNet101", "ResNet152"])
    parser.add_argument("--dataset_path", default="./dataset/")
    parser.add_argument("--csv_path", default="./csv_files/")
    parser.add_argument("--decoder_mode", default="up")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1773)
    parser.add_argument("--save", type=int, default=1)

    args = parser.parse_args()
    
    for arch in ["Resnet50", "DeiTSmall"]:
        args.arch = arch
        set_seed(args.seed)
        logger = Logger()
        logger(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger(f"device: {device}")

        trainer(args, device, logger)
