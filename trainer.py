import torch
from utils import linear_beta_scheduler, get_index_from_list
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="Simple Difussion",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "UNet",
    "dataset": "StanfordCars",
    "epochs": 10,
    }
)





class Trainer():
    def __init__(self, model, dataloader, loss, optimizer, batch_size, device="cuda", T=300) -> None:
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Define beta schedule    
        # Pre-calculate different terms for closed form
        self.T = T
        self.betas = linear_beta_scheduler(timesteps=self.T)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_difussion(self, x_0, t):
        #noise in the shape of the image
        noise = torch.randn_like(x_0)
        t_noise = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        _1_minus_t_noise = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # add mean + variance to image
        return t_noise.to(self.device)  * x_0.to(self.device) \
                + _1_minus_t_noise.to(self.device)  * noise.to(self.device),\
                noise.to(self.device)



    @torch.no_grad()
    def timestep(self, image, t):
        #This function
        #1.calls the model to predict the added noise
        #2.applies noise to the image, if t < 300

        betas_t = get_index_from_list(self.betas, t, image.shape)
        
        t_noise = get_index_from_list(self.sqrt_recip_alphas, t, image.shape)
        _1_minus_t_noise = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, image.shape)

        #current call to model
        model_predict = self.model(image, t)

        model_mean  = t_noise * ( image - betas_t * model_predict / _1_minus_t_noise)
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, image.shape)
        
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(image)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample_plot_image(self):
        # # Uncomment for notebook execution
        # # Sample noise
        # img_size = 64
        # img = torch.randn((1, 3, img_size, img_size), device=self.device)
        # plt.figure(figsize=(15,15))
        # plt.axis('off')
        # num_images = 10
        # stepsize = int(self.T/num_images)
        # for i in range(0, self.T)[::-1]:
        #     t = torch.full((1,), i, device=self.device, dtype=torch.long)
        #     img = self.timestep(img, t)
        #     if i % stepsize == 0:
        #         plt.subplot(1, num_images, i/stepsize+1)
        #         self.show_tensor_image(img.detach().cpu())
        #     plt.show()  
        pass



    def train(self, epochs=100):
        for epoch in range(1, epochs):
            for step, batch in enumerate(tqdm(self.dataloader)):
                self.optimizer.zero_grad()

                t = torch.randint(0, self.T, (self.batch_size,)).long().cuda()
                x_noisy, noise = self.forward_difussion(batch[0], t)
                noise_pred = self.model(x_noisy.cuda(), t)
                loss = self.loss(noise, noise_pred)
                loss.backward()
                self.optimizer.step()

                wandb.log({"loss": loss.item()})
                if step % 50 == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    self.sample_plot_image()