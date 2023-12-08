import os
import tensorflow as tf
import matplotlib.pyploy as plt 
import numpy as np
import logging
from tqdm import tqdm

class Diffusion:

    def __init__(
        self, 
        noise_steps=1000, 
        beta_start=1e-4, 
        beta_end=0.02, 
        img_size=256, 
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = beta_start # self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat =  self.alpha #torch.cumprod(self.alpha, dim=0)


    def prepare_noise_schedule(self):
        return np.linspace(
            self.beta_start, 
            self.beta_end, 
            self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = np.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = np.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = np.random.randn_like(x)
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return np.random.randint(
            low=1, 
            high=self.noise_steps, 
            size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()

        # initalize values for sampling
        x = np.random.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            t = (np.ones(n) * i).astype(np.int32)

            pred_noise = model(x, t)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]   

            noise = np.random.rand(*x.shape) if i > 1 else np.zeros_like(x)
            x = 1 / np.sqrt(alpha) * (x - ((1 - alpha) / (np.sqrt(1 - alpha_hat))) * pred_noise) + np.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).astype(np.int8)

        return x 