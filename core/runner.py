import os
import time
import json
import torch
import torchvision

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from core.data import MyDataset
from core.model import MyModel

class Runner():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_model(self):
        args = self.args
        device = self.device
        expr_root = f"{args.expr_root}/{args.trial_name}"
        
        os.makedirs(f"{expr_root}/checkpoints", exist_ok=True)
        os.makedirs(f"{expr_root}/training/losses", exist_ok=True)
        os.makedirs(f"{expr_root}/training/images", exist_ok=True)
        os.makedirs(f"{expr_root}/training/plots", exist_ok=True)
        os.makedirs(f"{expr_root}/validation", exist_ok=True)
        
        torch.manual_seed(args.seed)
        print(args)
        with open(f"{expr_root}/trial_args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)
            
        train_data = MyDataset(args.train_root)
        val_data   = MyDataset(args.val_root)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=1, shuffle=False)
        
        model = MyModel().to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_factor)
        
        if args.resume_epoch > 0:
            ckpt_path = f"{expr_root}/checkpoints/{args.resume_epoch}.pth"
            if os.path.exists(ckpt_path):
                data = torch.load(ckpt_path)
                model.load_state_dict(data["model_state"])
                optimizer.load_state_dict(data["optimizer_state"])
                scheduler.load_state_dict(data["scheduler_state"])
            else:
                print(f"Checkpoint not found at {ckpt_path}")
                return
            
        MSE = torch.nn.MSELoss()
        L1  = torch.nn.L1Loss()
        
        start_time = time.time()
        history  ={
            "train_loss" : [[], []],
            "val_loss" : [[], []]
        }
        for ep in range(args.resume_epoch+1, args.epochs+1):
            model.train()
            train_set = iter(train_loader)
            losses = {
                "loss" : 0.0
            }
            for data_point in train_set:
                data_point = data_point.to(device)
                
                pred = model(data_point)
                
                loss = MSE(data_point, pred)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses["loss"] += data_point.shape[0] * loss.item()
                
            losses["loss"] / len(train_data)
            loss = losses["loss"]
            cum_time = time.time() - start_time
            print(f"[{cum_time:.2f}] (TRAINING) Epoch {ep:04}/{args.epochs}: Loss [{loss:.6f}]")
            
            history["train_loss"][0].append(ep)
            history["train_loss"][1].append(loss)
            
            if ep % args.log_rate == 0:
                with open(f"{expr_root}/training/losses/losses_{ep}.json", "w") as f:
                    json.dump(losses, f, indent=4)
                
                plt.plot(history["train_loss"][0], history["train_loss"][1])
                plt.plot(history["val_loss"][0], history["val_loss"][1])
                plt.savefig(f"{expr_root}/training/plots/loss.png")
                plt.clf()
                
            if ep % args.val_rate == 0:
                with torch.no_grad():
                    model.eval()
                    val_set = iter(val_loader)
                    counter = 0
                    val_losses = {
                        "loss" : 0.0
                    }
                    for data_point in tqdm(val_set):
                        data_point = data_point.to(device)
                        
                        pred = model(data_point)
                        
                        loss = MSE(data_point, pred)
                        
                        val_losses["loss"] += loss.item()
                        
                        if counter < args.val_limit:
                            counter += 1
                            # save result
                            
                    val_losses["loss"] /= len(val_data)
                    loss = val_losses["loss"]
                    cum_time = time.time() - start_time
                    print(f"[{cum_time:.2f}] (TRAINING) Epoch {ep:04}/{args.epochs}: Loss [{loss:.6f}]")

                    with open(f"{expr_root}/validation/losses_{ep}.json", "w") as f:
                        json.dump(val_losses, f, indent=4)
                        
                    history["val_loss"][0].append(ep)
                    history["val_loss"][1].append(val_losses["loss"])
            
            if ep % args.save_rate == 0:
                save_path = f"{expr_root}/checkpoints/{ep}.pth"
                print(f"Saving checkpoint to {save_path}")
                torch.save({
                    "epoch" : ep,
                    "model_state" : model.state_dict(),
                    "optimizer_state" : optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict()
                }, save_path)
                
            if ep % args.lr_decay_rate == 0:
                scheduler.step()
                print(f"Encoder LR: {scheduler.get_last_lr()[0]}")
                    