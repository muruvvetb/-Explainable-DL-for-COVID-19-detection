import torch
import random
import numpy as np
import time
import os
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, extension = ""):
        self.base = "results"
        os.makedirs(self.base, exist_ok=True)
        self.folder_name = time.strftime("%Y %m %d - %H %M %S")
        self.path = f"{self.base}/{self.folder_name}_{extension}"
        os.makedirs(self.path, exist_ok=True)
        self.log_file_name = self.path + "/log.txt"
        self.log_file = open(self.log_file_name, "w")

    def __call__(self, item):
        print(item)
        print(item, file=self.log_file)

    def write(self, data):
        os.makedirs(f"{self.path}/logs", exist_ok=True)
        for mode in data:
            for score in data[mode]:
                f = open(f"{self.path}/logs/{mode}_{score}.csv", 'w')
                for i, j in zip(data[mode][score]["idx"], data[mode][score][score]):
                    f.write(f"{i},{j}\n")
                f.close()

    def plot(self, data):
        f, ax = plt.subplots(3, 1)
        ax[0].set_title("Loss")
        ax[0].plot(data["train"]["loss"]["idx"], data["train"]["loss"]["loss"], label="Train")
        ax[0].plot(data["val"]["loss"]["idx"], data["val"]["loss"]["loss"], label="Val")
        ax[0].legend()

        ax[1].set_title("Accuracies")
        ax[1].plot(data["train"]["acc"]["idx"], data["train"]["acc"]["acc"], label="Train")
        ax[1].plot(data["val"]["acc"]["idx"], data["val"]["acc"]["acc"], label="Val")
        ax[1].legend()

        ax[2].set_title("Test Loss and Accuracies")
        ax[2].plot(data["test"]["loss"]["idx"], data["test"]["loss"]["loss"], label="Loss")
        ax[2].plot(data["test"]["acc"]["idx"], data["test"]["acc"]["acc"], label="Acc")
        ax[2].legend()

        plt.tight_layout()
        plt.savefig(f"{self.path}/log.png")
        plt.savefig(f"{self.path}/log.pdf")
        plt.close()


def calculate_AAPPC(model, loader, device):
    model.eval()
    original_images = []
    list_explanations = []
    total_item = 0
    with torch.no_grad():
        for data, target in loader:
            if target[target == 1].shape[0] != 0:
                data = torch.index_select(data, 0, (target == 1).nonzero().reshape(-1))
                target = torch.index_select(target, 0, (target == 1).nonzero().reshape(-1))
                data, target = data.to(device), target.to(device)

                _, _, explanations = model.predict(data)
                original_images.extend(data.cpu())
                list_explanations.extend(explanations.cpu())
                total_item += data.shape[0]

    original_images = torch.stack(original_images)
    list_explanations = torch.stack(list_explanations)
    aappc = 0.0
    pertubation_ratios = np.arange(0.1, 1.01, 0.1)
    for i, image in enumerate(list_explanations):
        with torch.no_grad():
            p_0 = model.get_probabilities(original_images[i].unsqueeze(0).to(device))[0][1].item()
            for pertubation_ratio in pertubation_ratios:
                np_image = np.array(image)
                mask = np_image < pertubation_ratio
                np_original_image = np.array(original_images[i])
                for channel in range(np_original_image.shape[0]):
                    np_original_image[channel][mask[0]] = 0
                p_x = model.get_probabilities(torch.from_numpy(np_original_image).unsqueeze(0).to(device))[0][1].item()

                aappc += p_0 - p_x

    aappc /= 10.0

    return aappc
