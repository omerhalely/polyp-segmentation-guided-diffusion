import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import numpy as np
import os
from tqdm import tqdm
import imageio
import warnings


class ModelParameters:
    def __init__(self, model_name, batch_size, lr, epochs, num_training_steps, beta_start, beta_end, num_testing_steps,
                 guided, apply_cfg, cfg_prob):
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_training_steps = num_training_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_testing_steps = num_testing_steps
        self.guided = guided
        self.apply_cfg = apply_cfg
        self.cfg_prob = cfg_prob

    def write_parameters(self, average_loss):
        model_file_path = os.path.join(os.getcwd(), "saved_models", self.model_name, "Model Summary.log")
        with open(model_file_path, "w") as file:
            file.write(f"Model Name: {self.model_name}\n")
            file.write(f"Average Loss: {average_loss}\n")
            file.write(f"Batch Size: {self.batch_size}\n")
            file.write(f"Learning Rate: {self.lr}\n")
            file.write(f"Epochs: {self.epochs}\n")
            file.write(f"Num Training Steps: {self.num_training_steps}\n")
            file.write(f"Num Testing Steps: {self.num_testing_steps}\n")
            file.write(f"Beta Start: {self.beta_start}\n")
            file.write(f"Beta End: {self.beta_end}\n")
            file.write(f"Guided: {self.guided}\n")
            file.write(f"Apply CFG: {self.apply_cfg}\n")
            file.write(f"CFG Probability: {self.cfg_prob}")


def sample(model, vae, sampler, image, num_training_steps, num_testing_steps, device, cfg_scale, guided):
    noise = torch.FloatTensor(torch.randn(image.shape, dtype=torch.float32)).to(device)
    empty_condition = torch.zeros(image.shape, device=device)

    noise_images = []
    for timestep in range(num_training_steps - 1, 0, -int(num_training_steps / num_testing_steps)):
        with torch.no_grad():
            t = torch.tensor([timestep], device=device)
            noise_prediction = (1 + cfg_scale) * model(noise, t, image) - \
                               cfg_scale * model(noise, t, empty_condition)
            noise = sampler.step(noise_prediction, int(timestep), noise, return_dict=False)[0]
            noise_images.append(noise)

    with torch.no_grad():
        timestep = 0
        t = torch.unsqueeze(torch.tensor(timestep, device=device), dim=0)
        noise_prediction = (1 + cfg_scale) * model(noise, t, image) - cfg_scale * model(noise, t, empty_condition)
        noise = sampler.step(noise_prediction, int(timestep), noise, return_dict=False)[0]
        noise_images.append(noise)
        if guided:
            noise = noise / 0.18125
            prediction = vae.decode(noise.to(device)).sample
        else:
            prediction = noise
    return prediction, noise_images


def create_GIF(vae, images_list, path, device):
    print("Creating GIF")
    directory = os.path.join(os.getcwd(), "GIF images")
    if os.path.exists(directory):
        delete_dir(directory)
        os.mkdir(directory)
    else:
        os.mkdir(directory)

    images = []
    for i in tqdm(range(len(images_list))):
        image_path = os.path.join(os.getcwd(), "GIF images", f"{i}.png")

        noisy_image = images_list[i] / 0.18125
        prediction = vae.decode(noisy_image.to(device)).sample
        prediction = prediction[0]
        prediction = torch.sqrt(prediction[0, :, :] ** 2 + prediction[1, :, :] ** 2 + prediction[2, :, :] ** 2)
        prediction[prediction < 0.5] = 0
        prediction[prediction >= 0.5] = 1
        prediction = prediction.cpu().detach()

        plt.title(f"Segmentation Prediction - Iteration {i}/{len(images_list) - 1}")
        plt.imshow(prediction)
        plt.savefig(image_path)
        plt.close()

        images.append(imageio.imread(image_path))

    imageio.mimsave(path, images, duration=1)
    print(f"Saved GIF to {path}")
    delete_dir(directory)


def delete_dir(path):
    if os.path.exists(path):
        file_list = os.listdir(path)
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    os.rmdir(path)

def create_cfg_mask(shape, cfg_prob, gpu_id):
    cfg_mask = torch.zeros(shape[1:]) if np.random.random() < cfg_prob else torch.ones(shape[1:])
    cfg_mask = torch.unsqueeze(cfg_mask, dim=0)
    for i in range(shape[0] - 1):
        new_mask = torch.zeros(shape[1:]) if np.random.random() < cfg_prob else torch.ones(shape[1:])
        new_mask = torch.unsqueeze(new_mask, dim=0)
        cfg_mask = torch.cat((cfg_mask, new_mask), dim=0)
    cfg_mask = cfg_mask.to(gpu_id)
    return cfg_mask


def setup(rank, world_size):
    """
    Sets up the process group and configuration for PyTorch Distributed Data Parallelism
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
