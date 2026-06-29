import math
import os
from datetime import datetime
from abc import ABC

import numpy
import torch
import matplotlib.pyplot as plt

def get_2d_dataset(n_points, dataset_name):
    if dataset_name == "spiral":
        return get_spiral_dataset(n_points)
    elif dataset_name in ("uniform", "normal"):
        return get_random_dataset(n_points, dist=dataset_name)
        
    raise NotImplementedError(f"Unspecified dataset_name {dataset_name}.")

def get_spiral_dataset(n_points, noise_std=0.5, turns=2.0, rotation=0.0):
    t = torch.rand(n_points).sqrt()
    max_theta = turns * 2 * math.pi
    theta = t * max_theta
    theta_rotated = theta + rotation
    
    r = theta
    x = r * torch.cos(theta_rotated) + torch.randn(n_points) * noise_std
    y = r * torch.sin(theta_rotated) + torch.randn(n_points) * noise_std
    # x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    # y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)    
    
    data = torch.stack([x, y], dim=1)   # (N, 2)
    # data = torch.stack([x_norm, y_norm], dim=1)   # (N, 2)
    data = data.unsqueeze(-1)           # (N, 2, 1)
    
    return data

def get_random_dataset(n_points, dist="uniform", spiral_turns=2.0):
    spiral_max_radius = spiral_turns * 2 * math.pi
    
    if dist == "uniform":
        data = torch.rand((n_points, 2)) * (2 * spiral_max_radius) - spiral_max_radius
    elif dist == "normal":
        std_dev = spiral_max_radius / 3.0
        data = torch.randn((n_points, 2)) * std_dev
    else:
        raise NotImplementedError(f"Unspecified distribution : {dist}")
    
    data = data.unsqueeze(-1)
    return data


def save_2d_dataset_image(data, img_size=8, save_dir=None, file_name=None):    
    plt.figure(figsize=(img_size, img_size))
    if isinstance(data, dict):
        for i, (dataset_name, val) in enumerate(data.items()):
            cmap = plt.get_cmap('tab10')
            val = val.detach().cpu().numpy()
            plt.scatter(val[:, 0], val[:, 1], color=cmap(i%cmap.N), s=5, alpha=0.7, label=dataset_name) 
    elif isinstance(data, torch.Tensor) or isinstance(data, numpy.ndarray):
        plt.scatter(data[:, 0], data[:, 1], c='#1f77b4', s=5, alpha=0.7) 
    else:
        raise TypeError(f"Unexpected datatype of data : {type(data)}")
        
    if isinstance(data, dict):
        plt.legend(loc='best')
    
    plt.axis('equal') 
    plt.axis('off')
    # plt.title("Single Continuous Spiral Dataset")
    
    if save_dir is None:
        img_dir = os.path.join("results", "dataset")
        img_dir = os.path.join(img_dir, "get_spiral_dataset")
        os.makedirs(img_dir, exist_ok=True)
    else:
        assert os.path.exists(save_dir), f"save_dir(={save_dir}) does not exists."
        img_dir = save_dir
    
    if file_name is None:
        now_str = datetime.now().strftime("%y%m%d_%H%M")
        file_path = os.path.join(img_dir, f"two_dimensional-{now_str}.png")
    else:
        file_path = os.path.join(img_dir, f"{file_name}.png")
        
    plt.savefig(file_path)
    plt.close()




if __name__ == '__main__':
    n_points = 3000
    spiral_data1 = get_spiral_dataset(n_points, turns=2.0)
    spiral_data2 = get_spiral_dataset(n_points, turns=2.0, rotation=math.pi)
    uniform_data = get_random_dataset(n_points, "uniform")
    normal_data = get_random_dataset(n_points, "normal")
    data_dict = {
        "spiral1": spiral_data1,
        "spiral2": spiral_data2,
        "uniform": uniform_data,
        # "normal": normal_data,
    }    
    save_dir = "results/test/data/two_dimensional/"
    os.makedirs(save_dir, exist_ok=True)
    file_name = "spiral_uniform_normal_gen_test.png"
    save_2d_dataset_image(data_dict, save_dir=save_dir, file_name=file_name)