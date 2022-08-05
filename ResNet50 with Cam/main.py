import torch
import os
import numpy as np
import argparse
import time
import json
from utils import set_seed, Logger
from dataprep import get_dataloader
import torchvision.models as models
from tqdm import tqdm


def train(model, loss_func, train_loader, epoch, opt, device):
    model.train()

    total_loss = 0.0
    num_true = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        output = model(data)
        loss = loss_func(output, target)
        _, prediction = torch.max(output.data, 1)
        loss.backward()

        opt.step()
        total_loss += loss.item() * data.size(0)
        num_true += torch.sum(prediction == target.data)

    loss = total_loss / len(train_loader.dataset)
    num_true = np.squeeze(num_true.numpy()) if not 'cuda' else np.squeeze(num_true.cpu().numpy())
    accuracy = num_true / len(train_loader.dataset)

    return loss, accuracy


def eval(model, loss_func, loader, device):
    model.eval()
    total_loss = 0.0
    num_true = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            _, prediction = torch.max(output.data, 1)

            total_loss += loss.item() * data.size(0)
            num_true += torch.sum(prediction == target.data)

        loss = total_loss / len(loader.dataset)
        accuracy = num_true / len(loader.dataset)

    return loss, accuracy


def trainer(args, device, logger):
    logger(f'Train started at: {time.strftime("%Y %m %d - %H %M %S")}')
    start = time.time()
    batch_size = args.batch_size
    train_loader, val_loader, test_loader = get_dataloader(batch_size, args.dataset_path, args.csv_path, image_size=224,
                                                           mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                                           device=device)
    num_classes = len(set(train_loader.dataset.target_classes))
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, num_classes)
    '''if args.arch == "DeiTSmall":
        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True)
        model.head.out_features = num_classes'''

    logger(f"model: {model}")
    lrate = 1e-5
    logger(f"lrate: {lrate}")
    opt = torch.optim.Adam(model.parameters(), lr=lrate)
    logger(f"opt: {opt}")
    scheduler = torch.optim.lr_scheduler.StepLR(opt, args.max_epochs // 10, gamma=0.2)
    logger(f"scheduler: {scheduler}")
    criterion = torch.nn.CrossEntropyLoss()
    logger(f"Loss function: {criterion}")

    model.to(device)
    results = {
        "train": {"loss": {"idx": [], "loss": []}, "acc": {"idx": [], "acc": []}},
        "val": {"loss": {"idx": [], "loss": []}, "acc": {"idx": [], "acc": []}},
        "test": {"loss": {"idx": [], "loss": []}, "acc": {"idx": [], "acc": []}}
    }
    progress_bar = tqdm(range(args.max_epochs), position=0)
    best_val_acc = -1
    for epoch in progress_bar:
        tr_loss, tr_accuracy = train(model, criterion, train_loader, epoch, opt, device)
        results["train"]["loss"]["idx"].append(epoch)
        results["train"]["loss"]["loss"].append(tr_loss)

        results["train"]["acc"]["idx"].append(epoch)
        results["train"]["acc"]["acc"].append(float(tr_accuracy))

        val_loss, val_accuracy = eval(model, criterion, val_loader, device)
        results["val"]["loss"]["idx"].append(epoch)
        results["val"]["loss"]["loss"].append(val_loss)

        results["val"]["acc"]["idx"].append(epoch)
        results["val"]["acc"]["acc"].append(float(val_accuracy))

        to_print = f'Epoch: {epoch}: || Train Loss: {tr_loss:.3f}, Train Acc: {tr_accuracy:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.3f}'

        if epoch % 5 == 0 or epoch == args.max_epochs - 1:
            test_loss, test_accuracy = eval(model, criterion, test_loader, device)
            results["test"]["loss"]["idx"].append(epoch)
            results["test"]["loss"]["loss"].append(test_loss)

            results["test"]["acc"]["idx"].append(epoch)
            results["test"]["acc"]["acc"].append(float(test_accuracy))

            to_print += f', Test Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.3f}'

            logger.plot(results)

        progress_bar.set_description(desc=to_print)
        logger(to_print)

        if best_val_acc < val_accuracy:
            torch.save(model.state_dict(), logger.path + "/best_model.pt")
            best_val_acc = val_accuracy

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
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1773)
    parser.add_argument("--save", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)
    logger = Logger()
    logger(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger(f"device: {device}")

    trainer(args, device, logger)
