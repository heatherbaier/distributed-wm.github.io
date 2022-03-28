from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, models
from collections import OrderedDict
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
import torch.nn as nn
import pandas as pd
import numpy as np
import tempfile
import random
import torch
import sys
import os
import gc


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"


tfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



# @record
def load_ddp_state(state_dict):

    r18 = models.resnet18()

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation[key]
        new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict


class Dataloader():
    
    def __init__(self, country, imagery_direc, scores_df, split, batch_size, tfs = None):
        """
        Arguments:
            country: one of ['mex', 'slv', 'peru', 'phl']
            imagery_direc: path to folder containing school imagery
            scores_df: path to CSV file with school IDs and test scroes
            split: train/test split, should be between .01 and 1, recommended is between .65 and .8
            batch_size: number of images in a batch
        """
        self.country = country
        self.imagery_direc = imagery_direc
        self.imagery = os.listdir(self.imagery_direc)
        self.imagery = [i for i in self.imagery if self.country in i]
        self.scores_df = pd.read_csv(scores_df)
        self.scores_df = self.scores_df[self.scores_df['country'] == self.country]
        self.split = split
        self.batch_size = batch_size
        
#         print(self.imagery)
        
        if tfs is None:
            self.tfs = transforms.ToTensor()
        else:
            self.tfs = tfs
        
        # Load the data into a list with the format [(school_image, school_test_score), ...]
        self.data = self.load_data()
        
        # Split the data into training and validation sets
        self.train, self.val = self.test_train_split()
        
        
    def load_data(self):
        """
        Load the imagery into a list in the format: [(imager_tensor, test_score), ...]
        """
        data = []
        for col, row in self.scores_df[0:50].iterrows():
#             print(row)
            school_id = str(row.school_id)
            test_score = row.scaled_score
            impath = [i for i in self.imagery if school_id in i]
            if len(impath) > 0:
#                 print(impath)
                image = np.array(Image.open(self.imagery_direc + impath[0]))
                image = self.tfs(image)
                data.append((image, test_score))
        return data
                
        
    def test_train_split(self):
        """
        Split the data into 4 parts:
            x_train: imagery we use to train the model
            y_train: test_scores matched to the iamgery in x_train
            x_val: imagery used to test the model
            y_val: test_scores matched to the iamgery in x_val
        """
        train_num = int(len(self.data) * self.split)
        val_num = int(len(self.data) - train_num)

        all_indices = list(range(0, len(self.data)))
        train_indices = random.sample(range(len(self.data)), train_num)
        val_indices = list(np.setdiff1d(all_indices, train_indices))

        x_train, x_val = [self.data[i][0] for i in train_indices], [self.data[i][0] for i in val_indices]
        y_train, y_val = [self.data[i][1] for i in train_indices], [self.data[i][1] for i in val_indices]
        
        train = [(k,v) for k,v in zip(x_train, y_train)]
        val = [(k,v) for k,v in zip(x_val, y_val)]

        # Load the data into pytorch's Dataloader format
        
#         print(train)
        
        train = torch.utils.data.DataLoader(train, batch_size = self.batch_size, shuffle = True, drop_last = True)
        val = torch.utils.data.DataLoader(val, batch_size = self.batch_size, shuffle = True, drop_last = True)

        return train, val
        
        
        
def setup(rank, world_size):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12369'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    
COUNTRY = "mex"
BATCH_SIZE = 4
SPLIT = .75
IMAGERY_DIREC = "../../CCI/hmbaier/"
SCORES_DF = "../../CCI/hmbaier/cci_example.csv"


# We'll use the to keep track of our training stastics (i.e. running training loss, running validation loss, etc...)
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


def main(rank, world_size):
    
    setup(rank, world_size)
    
    print("here! in rank: ", rank, " with world size: ", world_size)
        
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])    
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.001)
            
#     imagery = get_imagery_list("../../../../../heather_data/temporal_features/jsons/", rank, world_size)
        
    data = Dataloader(country = COUNTRY, 
                      imagery_direc = IMAGERY_DIREC, 
                      scores_df = SCORES_DF,
                      split = SPLIT,
                      batch_size = BATCH_SIZE,
                      tfs = tfs)
    train_dl, val_dl = data.train, data.val

    train_tracker, val_tracker = AverageMeter(), AverageMeter()

    for epoch in range(0, 10):

        train_tracker.reset()
        val_tracker.reset()
        
        ddp_model.train()

        ######################################################
        # Train!
        ######################################################
        for (input, target) in train_dl:

            optimizer.zero_grad()
            
            input, target = torch.tensor(input, dtype = torch.float32).to(rank), torch.tensor(target, dtype = torch.float32).to(rank)
    
            output = ddp_model(input.to(rank))

            loss = criterion(output, target)
            
#             print("Rank: ", rank, "  Loss: ", loss.item())
                        
            train_tracker.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        with open(f"./records/{str(rank)}.txt", "w") as f:
            f.write(str(train_tracker.avg))
                
#         print("done with epoch in rank: ", rank)
            
        dist.barrier()
        
        if rank == 0:
            
            mean = []
            for i in os.listdir("./records/"):
                if "ipynb" not in i:
                    with open("./records/" + i, "r") as f:
                        mean.append(float(f.read()))
            
            print("Epoch: ", epoch)
            print(" Training MAE: ", np.average(mean))
            
        mname = "./models/model_epoch" + str(epoch) + ".torch"

        if rank == 1:

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                    },  mname)

        dist.barrier()

        ######################################################
        # Valdate!
        ######################################################

        weights = load_ddp_state(ddp_model.state_dict())
        model.load_state_dict(weights)
        model.eval()

        del weights
        gc.collect()

        with torch.no_grad():

            for (input, target) in val_dl:

                optimizer.zero_grad()

#                 input = input.permute(0,3,1,2)
#                 target = target.view(-1, 1)

                input, target = torch.tensor(input, dtype = torch.float32).to(rank), torch.tensor(target, dtype = torch.float32).to(rank)

                output = model(input)

                loss = criterion(output, target).item()
                val_tracker.update(loss)

        with open(f"./records/{str(rank)}.txt", "w") as f:
            f.write(str(val_tracker.avg))
                            
        dist.barrier()
        
        if rank == 0:
            
            mean = []
            for i in os.listdir("./records/"):
                if "ipynb" not in i:
                    with open("./records/" + i, "r") as f:
                        mean.append(float(f.read()))
            
            print(" Validation MAE: ", np.average(mean), "\n")
            
            
if __name__ == "__main__":
    
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    
    
