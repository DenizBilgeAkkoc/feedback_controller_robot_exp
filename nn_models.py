import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math


class CustomLoss(nn.Module):
    def __init__(self, T=None, C=None, D=None, E=None, use_relu_loss=False):
        super(CustomLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.T = T
        self.C = C
        self.D = D
        self.E = E
        self.use_relu_loss = use_relu_loss

    def forward(self, action_predicted, action_ground_truth, x_des, x_t, t):
        t = t / self.T
        loss_torques = self.criterion(action_predicted, action_ground_truth)

        mse_latent = torch.mean((x_des - x_t).pow(2), dim=1) #(batch_size, 1)
        
        if self.use_relu_loss:
            condition_mask = t < 0.4
            scalar = (~condition_mask).float()
        else:
            exp = torch.exp(self.D * (t - self.E)) #(batch_size, 1)
            scalar = self.C * exp
            # scalar = exp
            # scalar = 1.0

        scaled_mse_latent = scalar * mse_latent
        average_scaled_mse_latent = torch.mean(scaled_mse_latent) #(1, 1)

        # print("loss_torques", loss_torques, "C*av_ms_la", self.C*average_scaled_mse_latent)

        # return loss_torques, loss_torques
        return (loss_torques + (average_scaled_mse_latent)), loss_torques
        # return (loss_torques + (self.C*average_scaled_mse_latent)), loss_torques
    

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=0), #120x120 => 60x60
            nn.ReLU(True),
            nn.Conv2d(8, 32, kernel_size=2, stride=2, padding=0), #60x60 => 30x30
            nn.ReLU(True),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #30x30 => 30x30
            # nn.BatchNorm2d(16),
            # nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0), #30x30 => 15x15
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0), #15x15 => 7x7
            nn.ReLU(True)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(7*7*32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)

            # nn.Linear(15*15*16, encoded_space_dim),
            # nn.ReLU(True),
            # nn.Linear(256, encoded_space_dim)
        )
        
    def forward(self, x):
        # print(x.shape)
        x = self.encoder_cnn(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.encoder_lin(x)
        # print(x.shape)
        # print("-")
        return x
    

class TinyConvNet(nn.Module):
    def __init__(self, encoded_space_dim):
        super(TinyConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 30 * 30, encoded_space_dim)  
        # Assuming input image size is 120x120 after pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc(x)
        return x
    

class AlexNetPT(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # super(AlexNetPT, self).__init__()

        alexnet = models.alexnet(weights='DEFAULT')
        self.feature_extractor = alexnet.features
        print("featur extractor arch.:", self.feature_extractor)

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(256*5*9, 4),
            # nn.ReLU(True),
            # nn.Linear(1024, 128),
            # nn.ReLU(True),
            # nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        # print("Alexnet forward:")
        # print("input shape", x.shape)
        x = self.feature_extractor(x)
        # print("feature extractor shape", x.shape)
        x = self.flatten(x)
        # print("flatten shape", x.shape)
        x = self.encoder_lin(x)
        return x


class MLP_3L(nn.Module):
    def __init__(self, inp_dim, lat_dim_1, lat_dim_2, out_dim):
        super().__init__()
        # super(MLP, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(inp_dim, lat_dim_1),
            nn.ReLU(True),
            nn.Linear(lat_dim_1, lat_dim_2),
            nn.ReLU(True),
            nn.Linear(lat_dim_2, lat_dim_2),
            nn.ReLU(True),
            nn.Linear(lat_dim_2, out_dim),
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class GeneralModel(nn.Module):
    def __init__(self, encoded_space_dim, target_dim, action_dim, use_image):
        super().__init__()
        # super(GeneralModel, self).__init__()

        # self.alexnet = AlexNetPT(encoded_space_dim)
        # self.enc = TinyConvNet(encoded_space_dim)
        if use_image:
            # self.enc = Encoder(encoded_space_dim)
            self.enc = AlexNetPT(encoded_space_dim)
        else:
            self.enc = MLP_3L(target_dim, 384, 384, encoded_space_dim)

        # self.mlp_state = MLP(4, 128, encoded_space_dim)
        if use_image: controller_enc_dim = 512
        else: controller_enc_dim = 512
        self.mlp_controller = MLP_3L(encoded_space_dim, controller_enc_dim, 
                                  controller_enc_dim, action_dim)
        # Set the bias of the last linear layer to 0
        # self.mlp_controller.linear[2].bias.data.fill_(0.0)

    def forward(self, target_repr, state):
        # x = self.mlp_state(state)
        x = state

        # x_des = self.alexnet(img_tensor)
        x_des = self.enc(target_repr) # (batch_size, encoded_space_dim)
        
        # x_des = torch.squeeze(x_des, 0)
        diff = x_des - x

        acts_pred = self.mlp_controller(diff)
        #F.tanh
        return acts_pred, x_des, diff
    

class MLPBaseline(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        # super(MLP, self).__init__()

        # self.linear_1 = MLP_3L(inp_dim, 10, 24, 24)
        # self.linear_2 = MLP_3L(24, 24, 10, out_dim)
        # self.linear_2.linear[2].bias.data.fill_(0.0)

        self.linear = nn.Sequential(
            nn.Linear(inp_dim, 10),
            nn.ReLU(True),
            nn.Linear(10, 10),
            nn.ReLU(True),
            nn.Linear(10, 11),
            nn.ReLU(True),
            nn.Linear(11, 11),
            nn.ReLU(True),
            nn.Linear(11, 10),
            nn.ReLU(True),
            nn.Linear(10, out_dim),
        )
        # self.linear.linear[5].bias.data.fill_(0.0)

    def forward(self, x):
        x = self.linear(x)
        acts_pred = F.tanh(x)
        return acts_pred

        # x = self.linear_1(x)
        # acts_pred = F.tanh(self.linear_2(x))
        # return acts_pred