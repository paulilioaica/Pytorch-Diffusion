from trainer import Trainer
from dataloader import load_transformed_dataset
from torch.utils.data import DataLoader
from model import SimpleUnet
from loss import CustomLoss
from torch.optim import Adam
import argparse

parser = argparse.ArgumentParser(description='Train Simple Unet model')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help='device to train on (cuda or cpu)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')

args = parser.parse_args()


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)

model = SimpleUnet()
loss = CustomLoss()
optimizer = Adam(model.parameters(), lr=args.lr)

trainer = Trainer(model=model, dataloader=dataloader, loss=loss, optimizer=optimizer, 
                  device=args.device, batch_size=args.batch_size)

trainer.train(epochs=args.epochs)