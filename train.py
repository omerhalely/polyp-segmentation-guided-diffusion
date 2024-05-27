import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from models.DiT import DiT_models
from models.DiT_cross import DiT_cross_models
from diffusers.models import AutoencoderKL
from polyp_dataset import polyp_dataset
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from collections import OrderedDict
import argparse
from utils import *


torch.manual_seed(42)


class Trainer:
    def __init__(self, model, model_name, load_pretrained_model, gpu_id, data_path, criterion, batch_size, lr, epochs,
                 num_training_steps, beta_start, beta_end, num_testing_steps, guided, apply_cfg, cfg_prob, cfg_scale):
        self.model = model.to(gpu_id)
        self.model_name = model_name
        self.load_pretrained_model = load_pretrained_model
        self.gpu_id = gpu_id
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_training_steps = num_training_steps
        self.num_testing_steps = num_testing_steps
        self.guided = guided
        self.apply_cfg = apply_cfg
        self.cfg_prob = cfg_prob
        self.cfg_scale = cfg_scale
        self.data_path = data_path

        self.model_parameters = ModelParameters(model_name, batch_size, lr, epochs, num_training_steps, beta_start,
                                                beta_end, num_testing_steps, guided, apply_cfg, cfg_prob)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
        self.vae.to(self.gpu_id)

        if self.load_pretrained_model:
            self.load_model()

        self.train_dataset = polyp_dataset(
            data_path=data_path,
            mode="train",
            device=self.gpu_id
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                                           sampler=DistributedSampler(self.train_dataset, shuffle=True))

        self.sampler = DDPMScheduler(
            num_train_timesteps=self.num_training_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=False
        )

        self.writer = SummaryWriter(f"runs/{self.model_name}")

        self.ema = deepcopy(self.model).to(gpu_id)
        for param in self.ema.parameters():
            param.requires_grad = False

        self.model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.update_ema(self.ema, self.model.module, decay=0)
        self.ema.eval()

    def update_ema(self, ema_model, model, decay=0.9999):
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def train_one_epoch(self, epoch, images_list):
        self.model.train()

        dataloader = self.train_dataloader
        dataloader.sampler.set_epoch(epoch)

        total_loss = 0

        for batch_idx, (gt, image) in enumerate(dataloader):
            gt = gt.to(self.gpu_id)
            image = image.to(self.gpu_id)

            timestep = torch.randint(0, self.num_training_steps, (image.size(0), )).to(self.gpu_id)
            noise = torch.FloatTensor(torch.randn(gt.shape, dtype=torch.float32)).to(self.gpu_id)
            noisy_gt = self.sampler.add_noise(gt, noise, timestep)

            self.optimizer.zero_grad()

            if self.apply_cfg:
                image = image * create_cfg_mask(image.shape, self.cfg_prob, self.gpu_id)

            noise_prediction = self.model(noisy_gt.to(self.gpu_id), timestep, image.to(self.gpu_id))
            if batch_idx == 0:
                images_list.append(torch.unsqueeze(noise_prediction[0], dim=0))

            loss = self.criterion(noise_prediction.to(self.gpu_id), noise)

            loss.backward()
            self.optimizer.step()

            self.update_ema(self.ema, self.model.module)

            total_loss += loss.item()

        return total_loss / len(dataloader), images_list

    def train(self):
        print(f"Start Training {self.model_name}...")

        if not os.path.exists(os.path.join(os.getcwd(), "saved_models")):
            os.mkdir(os.path.join(os.getcwd(), "saved_models"))
        if not os.path.exists(os.path.join(os.getcwd(), "saved_models", self.model_name)):
            os.mkdir(os.path.join(os.getcwd(), "saved_models", self.model_name))

        images_list = []
        training_avg_loss = 0
        for epoch in range(self.epochs):
            print("-" * 40)

            training_avg_loss, images_list = self.train_one_epoch(epoch, images_list)

            print("-" * 40)
            print(f"| End of epoch {epoch} | Loss {training_avg_loss:.5f} |")

            self.writer.add_scalars(f"Loss/{self.model_name}", {"Train Loss": training_avg_loss}, epoch)
            if self.gpu_id == 0:
                self.save_model()

        # create_GIF(self.vae, images_list, os.path.join(os.getcwd(), "saved_models", self.model_name, "GIF.gif"), self.gpu_id)

        self.model_parameters.write_parameters(training_avg_loss)

    def save_model(self):
        checkpoint_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        state = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(state, checkpoint_path)

    def load_model(self):
        print(f"Loading model {self.model_name}...")
        model_path = os.path.join(os.getcwd(), "saved_models", self.model_name, f"{self.model_name}.pt")
        assert os.path.exists(model_path), f"Model {self.model_name}.pt does not exist."

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
        self.ema.load_state_dict(checkpoint["ema"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded Model Successfully.")


def main(rank: int, world_size: int, args, model_name: str):
    print(f"Detected {world_size} {'GPU' if world_size == 1 else 'GPUs'}")
    setup(rank, world_size)

    data_path = args.data_path
    model_name = model_name
    batch_size = args.batch_size
    lr = 1e-4
    beta_start = 10 ** -4
    beta_end = 2 * 10 ** -2
    epochs = args.epochs
    num_training_steps = 1000
    num_testing_steps = 100
    criterion = nn.MSELoss()
    apply_cfg = True
    cfg_prob = 0.1
    cfg_scale = 3
    guided = True
    load_pretrained_model = args.load_pretrained

    model_type = args.model
    model_names = {
        "DiT_XL2":"DiT-XL/2", "DiT_XL4":"DiT-XL/4", "DiT_XL8":"DiT-XL/8",
        "DiT_L2":"DiT-L/2", "DiT_L4":"DiT-L/4", "DiT_L8":"DiT-L/8",
        "DiT_B2":"DiT-B/2", "DiT_B4":"DiT-B/4", "DiT_B8":"DiT-B/8",
        "DiT_S2":"DiT-S/2", "DiT_S4":"DiT-S/4", "DiT_S8":"DiT-S/8",
    }
    if args.cross_model:
        model = DiT_cross_models[model_names[model_type]](in_channels=4, condition_channels=4, learn_sigma=False)
    else:
        model = DiT_models[model_names[model_type]](in_channels=4, condition_channels=4, learn_sigma=False)

    handler = Trainer(model=model,
                      model_name=model_name,
                      load_pretrained_model=load_pretrained_model,
                      gpu_id=rank,
                      data_path=data_path,
                      criterion=criterion,
                      batch_size=batch_size,
                      lr=lr,
                      epochs=epochs,
                      num_training_steps=num_training_steps,
                      beta_start=beta_start,
                      beta_end=beta_end,
                      num_testing_steps=num_testing_steps,
                      guided=guided,
                      apply_cfg=apply_cfg,
                      cfg_prob=cfg_prob,
                      cfg_scale=cfg_scale)

    print(f"Training Model: {model_name}\nModel Type: {model_names[model_type]}\nData Path: {data_path}\n"
          f"Batch Size: {batch_size}\nEpochs: {epochs}\nPretrained: {load_pretrained_model}\n"
          f"Cross Model: {args.cross_model}")

    handler.train()
    cleanup()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Did not find a GPU"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT_S8", choices=["DiT_XL2", "DiT_XL4", "DiT_XL8",
                                                                        "DiT_L2", "DiT_L4", "DiT_L8",
                                                                        "DiT_B2", "DiT_B4", "DiT_B8",
                                                                        "DiT_S2", "DiT_S4", "DiT_S8"])
    parser.add_argument("--data-path", type=str, default="./data/Kvasir-SEG")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--load-pretrained", type=bool, default=False)
    parser.add_argument("--cross-model", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--num-augmentations", type=int, default=8)


    args = parser.parse_args()
    args.cross_model = True if args.cross_model == "true" else False
    model_name = f"{args.model}_{'CROSS' if args.cross_model else ''}_{args.data_path.split('/')[-1]}_{args.epochs}_epochs_{args.batch_size}_batch_{args.num_augmentations}_augmentations"

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args, model_name,), nprocs=world_size)


