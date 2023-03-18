# 📊 PyTorch Diffusion Model for Stanford Cars Dataset 🚗 🚂


This project implements a simple minimalist diffusion model in PyTorch 🔥 and integrates with the W&B platform 💻 for experiment tracking and visualization 📈. The model is trained on the Stanford Cars dataset 🚙 and is designed for image generation tasks.



# 📈 What is a diffusion model?

A diffusion model is a type of generative model that learns to generate samples from a target distribution by gradually spreading probability mass from a simple base distribution through a series of diffusion steps. The diffusion process allows the model to capture complex dependencies between variables, making it well-suited to tasks such as image generation and denoising.

# 💻 Installation
To get started with this project, you can clone the repository and install the necessary dependencies using pip:


```git clone https://github.com/your_username/your_project.git
cd your_project
pip install -r requirements.txt
```

# 🔥 Usage
To train the diffusion model on the Stanford Cars dataset, simply run the following command:


```
python train.py --epochs 100 --lr 0.0005 --batch_size 16 --device cuda

```

This will start the training process and log metrics to W&B for easy tracking and visualization.
 

# 📈 Experiment Tracking
This project integrates with W&B for experiment tracking and visualization. You can view your experiment runs in the W&B dashboard 📊 by logging in with your W&B account credentials:

```wandb login
```

You can then view your experiment runs in the dashboard by navigating to the project page and selecting the relevant runs.


# 📚 References
If you're interested in learning more about diffusion models or PyTorch, you can check out the following resources:


[Diffusion Models](https://arxiv.org/abs/2105.05233)

[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

[Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

We hope you enjoy using this project! 😄
Feel free to clone and use your own dataset.