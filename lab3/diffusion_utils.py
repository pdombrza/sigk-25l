import torch
import numpy as np


class DDPMSampler:
    def __init__(self, beta_start: float = 1e-4, beta_end: float = 1e-2, n_timesteps: int = 1000):
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps, dtype=torch.float32)  # linear schedule
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.n_timesteps = n_timesteps
        self.timesteps = torch.from_numpy(np.arange(0, n_timesteps)[::-1].copy())

    @torch.no_grad()
    def add_noise(self, original_sample, timestep):  # forward process
        noise = torch.randn_like(original_sample)
        alpha_bars = self.alpha_bars.to(device=original_sample.device, dtype=original_sample.dtype)
        timestep = timestep.to(device=original_sample.device)
        sqrt_alpha_bars = torch.sqrt(alpha_bars[timestep])  # mean
        sqrt_one_minus_alpha_bars = torch.sqrt((1.0 - alpha_bars[timestep]))  # stdev
        noisy_samples = (sqrt_alpha_bars * original_sample) + (sqrt_one_minus_alpha_bars) * noise
        return noisy_samples, noise

    @torch.no_grad()
    def denoise_step(self, image, timestep, predicted_noise):
        device = image.device
        alpha_t = self.alphas[timestep].to(device=device)
        alpha_bar_t = self.alpha_bars[timestep].to(device=device)
        beta_t = self.betas[timestep].to(device=device)

        mean_image = torch.sqrt(1.0 / alpha_t) * (image - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t))

        alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
        beta_t_bar = (1 - alpha_hat_prev) / (1 - alpha_bar_t) * beta_t
        variance = torch.sqrt(beta_t_bar) * torch.randn(image.shape).to(device) if timestep > 0 else 0

        image_denoised = mean_image + variance
        return image_denoised
