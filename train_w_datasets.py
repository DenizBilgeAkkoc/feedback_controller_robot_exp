random_seed = 1

import os
import time
import csv
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader

from nn_models import GeneralModel
from nn_models import MLPBaseline
from nn_models import CustomLoss

# np.random.seed(random_seed)
# torch.manual_seed(random_seed)


class TrajectoryDataset(Dataset):
    def __init__(self, ds_root_dir, file_name, use_image=False):
        data_file_path = os.path.join(ds_root_dir, file_name)
        #self.trajectories = np.expand_dims(np.load(data_file_path)[0], axis=0)
        self.trajectories = np.load(data_file_path)#[0:4]

        self.ds_root_dir = ds_root_dir
        self.num_trajectories_in_dataset = self.trajectories.shape[0]
        self.num_steps = self.trajectories.shape[1]
        self.use_image = use_image
        print("trajectories.shape", self.trajectories.shape)
        if self.use_image:
            self.transform = transforms.Compose([
                # transforms.Resize([108, 171]),
                # transforms.RandomHorizontalFlip(), # Flip the data horizontally
                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])

    def __len__(self):
        return self.num_trajectories_in_dataset*self.num_steps
        # return 70*49

    def __getitem__(self, idx):
        traj_no = idx // self.num_steps
        step_no = idx % self.num_steps

        step = self.trajectories[traj_no, step_no, 0]
        state = self.trajectories[traj_no, step_no, 1:15]
        action = self.trajectories[traj_no, step_no, 15:22]

        if self.use_image:
            img_path = os.path.join(self.ds_root_dir,
                                    f"scenes_cropped/eps_{traj_no}/crpd_step_{step_no}.png")
            image = Image.open(img_path)
            # image = read_image(img_path)
            image = self.transform(image)
            target_repr = image
        else:
            target_pos = self.trajectories[traj_no, step_no, 22:]
            target_repr = target_pos

        # print("target_repr.shape", target_repr.shape)
        return step, state, target_repr, action*10.
    

def log_loss(n, val_loss_cutsom, val_loss_torques):
    global weights_storage_root_dir
    global train_loss_custom, train_loss_torques

    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'a') as f:
        writer = csv.writer(f)
        row = [n, train_loss_custom, train_loss_torques, val_loss_cutsom, 
               val_loss_torques]
        writer.writerow(row)

    print("===")
    print(f"Epoch: {n}:")
    print(f"train_loss_custom: {train_loss_custom}")
    print(f"train_loss torques: {train_loss_torques}")
    print(f"val_loss_cutsom: {val_loss_cutsom}")
    print(f"val_loss_torques: {val_loss_torques}")


def run_test(n):
    global val_dataloader, model, criterion, weights_storage_root_dir
    global val_losses_custom, val_losses_torques
    global use_custom_loss, use_baseline
    global train_info

    val_loss_cutsom = 0
    val_loss_torques = 0
    for i, batch_data in enumerate(val_dataloader):
        batch_step = batch_data[0].to(device).float()
        batch_state = batch_data[1].to(device).float()
        batch_target_repr = batch_data[2].to(device).float()
        batch_action = batch_data[3].to(device).float()

        model.eval()
        with torch.no_grad():
            if use_baseline:
                nn_input = torch.cat((batch_target_repr, batch_state), dim=1)
                batch_action_pred = model(nn_input)
            else:
                batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                                   batch_state)
            
        if use_custom_loss:
            loss_custom, loss_torques = criterion(batch_action_pred, batch_action, 
                                                batch_x_des, batch_state, 
                                                batch_step)
        else:
            loss_torques = criterion(batch_action_pred, batch_action)
            loss_custom = loss_torques

        val_loss_cutsom += loss_custom.item() * batch_state.size(0)
        val_loss_torques += loss_torques.item() * batch_state.size(0)

    val_loss_cutsom /= len(val_dataloader.dataset)
    val_losses_custom.append(val_loss_cutsom)
    val_loss_torques /= len(val_dataloader.dataset)
    val_losses_torques.append(val_loss_torques)

    act_abs_diff = torch.abs(batch_action_pred - batch_action)
    mean_abs_diff = torch.mean(act_abs_diff, dim=0)
    for k in range(7):
        train_info[f'mae_joint_{k+1}_val'].append(mean_abs_diff[k].item())

    log_loss(n, val_loss_cutsom, val_loss_torques)
    model.train()


def visualize_losses(train_losses, val_losses, n, file_name, title, fig):
    global validation_interval, weights_storage_root_dir
    # ax.clear()
    plt.figure(fig)
    plt.clf()
    plt.plot([-1] + list(range(0, n+1, validation_interval)), val_losses, 
             label='Validation Loss')
    plt.plot([-1] + list(range(len(train_losses)-1)), train_losses, 
             label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    # Update the plot
   # plt.draw()
   # plt.pause(0.001)  # Pause for a short time to update the plot
    loss_fig_path = os.path.join(weights_storage_root_dir, f"{file_name}")
    plt.savefig(loss_fig_path)
    # plt.close()


def create_csv_files(weights_storage_root_dir, selected_val_ind):
    file_path = os.path.join(weights_storage_root_dir, "input_independent_baseline.csv")
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["n"]
        for j in range(2):
            row.append(f"act_pred_{j}")
            row.append(f"act_pred_zero_{j}")
            row.append(f"act_tru_{j}")
        writer.writerow(row)

    loss_file_path = os.path.join(weights_storage_root_dir, "loss.csv")
    with open(loss_file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["n", "train_loss_custom", "train_loss_torques", "val_loss_cutsom", 
            "val_loss_torques"]
        writer.writerow(row)

    file_path = os.path.join(weights_storage_root_dir, "prediction_dynamics.csv")
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        row = ["n"]
        for val_idx in selected_val_ind:
            for j in range(2):
                row.append(f"act_pred_{val_idx}_{j}")
                row.append(f"act_tru_{val_idx}_{j}")
        writer.writerow(row)


def overfit_one_batch(batch_target_repr, batch_state, batch_action, batch_step):
    global model, optimizer, criterion
    global use_custom_loss
    
    counter = 0
    while True:
        optimizer.zero_grad()
        batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                            batch_state)
        if use_custom_loss:
            loss_custom, loss_torques = criterion(batch_action_pred, batch_action, 
                                                batch_x_des, batch_state, 
                                                batch_step)
        else:
            loss_torques = criterion(batch_action_pred, batch_action)
            loss_custom = loss_torques

        loss_custom.backward()
        optimizer.step()

        counter += 1
        print(f"Epoch {counter}, Training loss: {loss_custom.item() * batch_state.size(0)}")
        print(f"Epoch {counter}, Torques loss: {loss_torques.item() * batch_state.size(0)}")
        # print("batch_action_pred", batch_action_pred)
        # print("batch_action", batch_action)
        print("===")


def input_independent_baseline(dataset, selected_train_idx):
    global model, criterion, device
    global use_custom_loss, weights_save_root, n
    global use_baseline, use_image

    input_ind_data = dataset.__getitem__(selected_train_idx)
    batch_step = torch.from_numpy(np.array(input_ind_data[0])).unsqueeze_(dim=0).to(device).float()
    batch_state = torch.from_numpy(input_ind_data[1]).unsqueeze_(dim=0).to(device).float()
    batch_action = torch.from_numpy(input_ind_data[3]).unsqueeze_(dim=0).to(device).float()
    if use_image:
        batch_target_repr = input_ind_data[2].unsqueeze_(dim=0).to(device).float()
    else:
        batch_target_repr = torch.from_numpy(input_ind_data[2]).unsqueeze_(dim=0).to(device).float()

    batch_target_zeros = torch.zeros_like(batch_target_repr)
    batch_state_zeros = torch.zeros_like(batch_state)

    model.eval()
    with torch.no_grad():
        if use_baseline:
            nn_input = torch.cat((batch_target_repr, batch_state), dim=1)
            nn_input_zeros = torch.cat((batch_target_zeros, batch_state_zeros), dim=1)
            batch_action_pred = model(nn_input)
            batch_action_pred_zeros = model(nn_input_zeros)
        else:
            batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                                batch_state)
            batch_action_pred_zeros, batch_x_des_zeros, batch_diff = model(batch_target_zeros, 
                                                                        batch_state_zeros)
        
        actions = [n]
        for j in range(batch_action.shape[1]):
            actions.append(batch_action_pred[:, j].item())
            actions.append(batch_action_pred_zeros[:, j].item())
            actions.append(batch_action[:, j].item())

        with open(weights_save_root + "/input_independent_baseline.csv", 'a') as f:
            writer = csv.writer(f)
            row = actions
            writer.writerow(row)

    model.train()


def prediction_dynamics(dataset, selected_val_ind):
    global model, criterion, device
    global use_custom_loss, weights_save_root, n

    actions = [n]
    for i, val_idx in enumerate(selected_val_ind):
        val_data = dataset.__getitem__(val_idx)
        batch_step = torch.from_numpy(np.array(val_data[0])).unsqueeze_(dim=0).to(device).float()
        batch_state = torch.from_numpy(val_data[1]).unsqueeze_(dim=0).to(device).float()
        # batch_target_repr = torch.from_numpy(val_data[2]).unsqueeze_(dim=0).to(device).float()
        batch_target_repr = val_data[2].unsqueeze_(dim=0).to(device).float()
        batch_action = torch.from_numpy(val_data[3]).unsqueeze_(dim=0).to(device).float()

        model.eval()
        with torch.no_grad():
            batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                               batch_state)
            batch_action_pred = batch_action_pred.cpu().detach().numpy()
            batch_action = batch_action.cpu().detach().numpy()

            for j in range(batch_action.shape[1]):
                actions.append(batch_action_pred[:, j].item())
                actions.append(batch_action[:, j].item())

    with open(weights_save_root + "/prediction_dynamics.csv", 'a') as f:
        writer = csv.writer(f)
        row = actions
        writer.writerow(row)

    model.train()


def data_tester(train_dataloader, model):
    batch_data = next(iter(train_dataloader))
    batch_step = batch_data[0].to(device).float()
    batch_state = batch_data[1].to(device).float()
    batch_target_repr = batch_data[2].to(device).float()
    batch_action = batch_data[3].to(device).float()

    print(batch_target_repr.shape)
    img = batch_target_repr[0].detach().cpu().numpy().transpose((1, 2, 0))  # Rearrange dimensions to height x width x channels
    print("image shape", img.shape)
    # plt.imshow(img)
    # plt.show()

    batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                                    batch_state)

    raise Exception("Testing only...")


# Custom Loss:
num_steps = 50
T = num_steps - 1
C = 5 #1
D = 10 #5
E = 1

# General:
current_dir_path = os.path.dirname(__file__)
current_dir_path = Path(current_dir_path)
fbc_root_dir_path = current_dir_path.parent.absolute()
episodes_num_ds = 815
use_baseline = False
use_image = False
use_custom_loss = False
dataset_name = f"{episodes_num_ds}_trajs_static"

# Model:
encoded_space_dim = 14
target_dim = 3
action_dim = 7

# Training:
num_epochs = 2200 + 1 
batch_size = 32
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device", device)
# device = "cpu"
validation_interval = 50
num_trains = 1

train_info = dict()
for i in range(7):
    train_info[f'mae_joint_{i+1}_val'] = []
    train_info[f'mae_joint_{i+1}_train'] = []

for i_train in range(num_trains):
    fig_1 = plt.figure(figsize=(12.8, 9.6))
    fig_2 = plt.figure(figsize=(12.8, 9.6))

    if use_custom_loss: 
        model_name = f"cus_los_{C}_{D}_{E}"
        # model_name = f"cus_los_const_mse_st"
    else: model_name = "mse_los"
    if use_image: model_name += "|tar_img"
    else: model_name += "|tar_cart"
    if use_baseline: model_name += "|base"
    # model_name += "ARASHTEST"

    if use_baseline:
        model = MLPBaseline(inp_dim=6, out_dim=2)
    else:
        model = GeneralModel(encoded_space_dim=encoded_space_dim, target_dim=target_dim,
                             action_dim=action_dim, use_image=use_image)
    m = model.to(device)
    num_params = sum(p.numel() for p in m.parameters())/1e3
    model_name += f"|{num_params}K_params"

    # Dataset
    ds_root_dir = os.path.join(fbc_root_dir_path, 
                               f'neural_network/data/torobo/{dataset_name}')
    dataset = TrajectoryDataset(ds_root_dir, 'trajectories_normalized_110.npy', use_image)
    # train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_set, _ = torch.utils.data.random_split(dataset, [1.0, 0])
    _, val_set = torch.utils.data.random_split(dataset, [0, 1.0])
    # train_set, val_set = dataset, dataset

    # print(len(dataset), len(train_set), len(val_set))
    selected_val_ind = random.sample(val_set.indices, k=3)
    selected_train_idx = random.sample(train_set.indices, k=1)[0]

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Loss
    if use_custom_loss:
        criterion = CustomLoss(T, C, D, E)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # data_tester(train_dataloader, model)

    weights_storage_root_dir = os.path.join(current_dir_path, 
                                            f"weights/{dataset_name}|{model_name}/train_no_{i_train}")

    print(f"=== Train No: {i_train} ===")
    print(f"train num_samples:{len(train_dataloader.dataset)}")
    print(f"val num_samples:{len(val_dataloader.dataset)}")
    print("device:", device)
    print("dataset_name", dataset_name)
    print("model_name", model_name)
    print(num_params, 'K parameters')
    print("type_of_criterion", type(criterion))
    print("weights_storage_root_dir", weights_storage_root_dir)

    if not os.path.exists(weights_storage_root_dir):
        os.makedirs(weights_storage_root_dir)
    
    create_csv_files(weights_storage_root_dir, selected_val_ind)

    # while True:
    #     prediction_dynamics(dataset, selected_val_ind)

    train_loss_custom = 0
    train_loss_torques = 0
    train_losses_custom = [train_loss_custom]
    train_losses_torques = [train_loss_torques]
    val_losses_custom = []
    val_losses_torques = []

    for n in range(num_epochs):
        if n == 0:
            run_test(n)
            # pass

        start_time = time.time()
        train_loss_custom = 0
        train_loss_torques = 0
        model.train()
        for i, batch_data in enumerate(train_dataloader):
            batch_step = batch_data[0].to(device).float()
            batch_state = batch_data[1].to(device).float()
            batch_target_repr = batch_data[2].to(device).float()
            batch_action = batch_data[3].to(device).float()

            # overfit_one_batch(batch_target_repr, batch_state, batch_action, batch_step)

            optimizer.zero_grad()
            if use_baseline:
                nn_input = torch.cat((batch_target_repr, batch_state), dim=1)
                batch_action_pred = model(nn_input)
            else:
                batch_action_pred, batch_x_des, batch_diff = model(batch_target_repr, 
                                                                    batch_state)
            if use_custom_loss:
                loss_custom, loss_torques = criterion(batch_action_pred, batch_action, 
                                                    batch_x_des, batch_state, 
                                                    batch_step)
            else:
                loss_torques = criterion(batch_action_pred, batch_action)
                loss_custom = loss_torques

            loss_custom.backward()
            optimizer.step()

            train_loss_custom += loss_custom.item() * batch_state.size(0)
            train_loss_torques += loss_torques.item() * batch_state.size(0)
        
        train_loss_custom /= len(train_dataloader.dataset)
        train_losses_custom.append(train_loss_custom)
        train_loss_torques /= len(train_dataloader.dataset)
        train_losses_torques.append(train_loss_torques)

        end_time = time.time()
        if n % validation_interval == 0:
            epoch_time = end_time - start_time
            print(f"Last epoch taken time: {epoch_time:.3f} seconds")

            act_abs_diff = torch.abs(batch_action_pred - batch_action)
            mean_abs_diff = torch.mean(act_abs_diff, dim=0)
            for k in range(7):
                train_info[f'mae_joint_{k+1}_train'].append(mean_abs_diff[k].item())

            run_test(n)

            # input_independent_baseline(dataset, selected_train_idx)
            # prediction_dynamics(dataset, selected_val_ind)

            weight_file_path = os.path.join(weights_storage_root_dir, f"fbc_{n}.pth")
            torch.save(model.state_dict(), weight_file_path)

            info_file_path = os.path.join(weights_storage_root_dir, f"info.pickle")
            with open(info_file_path, 'wb') as handle:
                pickle.dump(train_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

            visualize_losses(train_losses_custom, val_losses_custom, n, 
                            f"loss_custom_{n//100}.png", "Custom Loss", fig_1)
            visualize_losses(train_losses_torques, val_losses_torques, n, 
                            f"loss_torques_{n//100}.png", "Torques Loss", fig_2)
            
    plt.close('all')

