
#######   disentangle=False ########

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
import random
import xlrd

BATCH_SIZE = 1

kernel_size = 3
filters = 8
stride = 2
intermediate_dim = 128
latent_dim = 24

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.flatten_c = nn.Flatten()
        self.fc_c1 = nn.Linear(in_features=1 * 1 * 1 * 1 * 1, out_features=filters)
        self.relu_c1 = nn.ReLU(inplace=True)
        self.fc_c2 = nn.Linear(in_features=filters, out_features=filters * 2)
        self.relu_c2 = nn.ReLU(inplace=True)
        self.fc_c3 = nn.Linear(in_features=filters * 2, out_features=filters * 4)
        self.relu_c3 = nn.ReLU(inplace=True)
        self.fc_c4 = nn.Linear(in_features=filters * 4, out_features=filters * 4)
        self.relu_c4 = nn.ReLU(inplace=True)
        # encoder
        self.conv_1 = nn.Conv3d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv2 = nn.ReLU(inplace=True)
        self.down1 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu_down1 = nn.ReLU(inplace=True)

        self.conv_3 = nn.Conv3d(in_channels=filters, out_channels=filters * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv4 = nn.ReLU(inplace=True)
        self.down2 = nn.Conv3d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu_down2 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(in_channels=filters * 2, out_channels=filters * 4, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv3d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_conv6 = nn.ReLU(inplace=True)
        self.down3 = nn.Conv3d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size, stride=stride, padding=1)
        self.relu_down3 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(in_features=filters * 4 * 18 * 24 * 20, out_features=intermediate_dim)
        self.relu_fc0 = nn.ReLU(inplace=True)
        ################################################################################################
        self.z_mean = nn.Linear(in_features=intermediate_dim, out_features=latent_dim)  #计算均值
        self.z_log_var = nn.Linear(in_features=intermediate_dim, out_features=latent_dim)  #计算方差

        self.fc1 = nn.Linear(in_features=latent_dim, out_features=intermediate_dim)
        self.fc2 = nn.Linear(in_features=intermediate_dim, out_features=filters*4 * 18 * 24 * 20)

        # decoder
        self.up1 = nn.ConvTranspose3d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size, stride=stride,padding=1, output_padding=1)
        self.relu_up1 = nn.ReLU(inplace=True)
        self.Dconv1 = nn.Conv3d(in_channels=filters * 4, out_channels=filters * 4, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_Dconv1 = nn.ReLU(inplace=True)
        self.Dconv2 = nn.Conv3d(in_channels=filters * 4, out_channels=filters * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_Dconv2 = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose3d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size, stride=stride,padding=1, output_padding=1)
        self.relu_up2 = nn.ReLU(inplace=True)
        self.Dconv3 = nn.Conv3d(in_channels=filters * 2, out_channels=filters * 2, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_Dconv3 = nn.ReLU(inplace=True)
        self.Dconv4 = nn.Conv3d(in_channels=filters * 2, out_channels=filters, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_Dconv4 = nn.ReLU(inplace=True)

        self.up3 = nn.ConvTranspose3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        self.relu_up3 = nn.ReLU(inplace=True)
        self.Dconv5 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,padding=1)
        self.relu_Dconv5 = nn.ReLU(inplace=True)
        self.Dconv6 = nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1, padding=1)
        self.relu_Dconv6 = nn.ReLU(inplace=True)
        # self.sigmoid = nn.Sigmoid()
        self.Pre_conv = nn.Conv3d(in_channels=filters, out_channels=1, kernel_size=1, stride=1, padding=0)

    def reparameterize(self, z_mean, z_log_var):
        eps = Variable(torch.randn(z_mean.size(0), z_mean.size(1))).to(device)
        z = z_mean + eps * torch.exp(z_log_var / 2)
        return z

    def FiLM_layer(self, feaMap, conda):
        batch, channel, depth, height, width = feaMap.size()
        self.fc = nn.Linear(in_features=1*channel, out_features=2*channel).to(device)
        film_params = self.fc(conda).to(device)
        film_params = torch.unsqueeze(film_params,-1)
        film_params = torch.unsqueeze(film_params, -1)
        film_params = torch.unsqueeze(film_params, -1)
        film_params = torch.Tensor.repeat(film_params, [1,1,depth, height, width])
        gammas = film_params[:,:channel,:,:,:]
        betas = film_params[:, channel:, :, :, :]
        output = (1 + gammas) * feaMap + betas
        return output

    def forward(self, img,age):
        # out1, out2 = self.encoder(x), self.encoder(x)
        #cat_input = torch.cat((x, y), 1)
        age = self.flatten_c(age)
        age_f1 = self.relu_c1(self.fc_c1(age))
        age_f2 = self.relu_c2(self.fc_c2(age_f1))
        age_f3 = self.relu_c3(self.fc_c3(age_f2))
        #age_f4 = self.relu_c4(self.fc_c4(age_f3))
        img_conv1 = self.relu_conv1(self.conv_1(img))
        img_age_f1 = self.FiLM_layer(img_conv1, age_f1)
        img_conv2 = self.relu_conv2(self.conv_2(img_age_f1))
        img_down1 = self.relu_down1(self.down1(img_conv2))
        img_conv3 = self.relu_conv3(self.conv_3(img_down1))
        img_age_f2 = self.FiLM_layer(img_conv3, age_f2)
        img_conv4 = self.relu_conv4(self.conv4(img_age_f2))
        img_down2 = self.relu_down2(self.down2(img_conv4))
        img_conv5 = self.relu_conv5(self.conv5(img_down2))
        img_age_f3 = self.FiLM_layer(img_conv5, age_f3)
        img_conv6 = self.relu_conv6(self.conv6(img_age_f3))
        img_down3 = self.relu_down3(self.down3(img_conv6))
        #img_age_f4 = self.FiLM_layer(img_down3, age_f4)
        img_flatten = self.flatten(img_down3)
        img_fc = self.relu_fc0(self.fc0(img_flatten))

        z_mean, z_log_var = self.z_mean(img_fc), self.z_log_var(img_fc)  # batch_s, latent，计算均值和方差
        z = self.reparameterize(z_mean, z_log_var)  # batch_s, latent
        # out3 = self.fc2(z).view(z.size(0), 64, 7, 7)  # batch_s, 64, 7, 7

        out3 = self.fc1(z)
        out4 = self.fc2(F.relu(out3))  # batch_s, 64, 7, 7

        img_up1 = self.relu_up1(self.up1(F.relu(out4).view(img_fc.size(0), filters*4,18,24,20)))
        img_Dconv1 = self.relu_Dconv1(self.Dconv1(img_up1))
        img_age_Df1 = self.FiLM_layer(img_Dconv1, age_f3)
        img_Dconv2 = self.relu_Dconv2(self.Dconv2(img_age_Df1))

        img_up2 = self.relu_up2(self.up2(img_Dconv2))
        img_Dconv3 = self.relu_Dconv3(self.Dconv3(img_up2))
        img_age_Df2 = self.FiLM_layer(img_Dconv3, age_f2)
        img_Dconv4 = self.relu_Dconv4(self.Dconv4(img_age_Df2))

        img_up3 = self.relu_up3(self.up3(img_Dconv4))
        img_Dconv5 = self.relu_Dconv5(self.Dconv5(img_up3))
        img_age_Df3 = self.FiLM_layer(img_Dconv5, age_f1)
        img_Dconv6 = self.relu_Dconv6(self.Dconv6(img_age_Df3))
        out_decoder = self.Pre_conv(img_Dconv6)

        return out_decoder, z_mean, z_log_var, z

def loss_func(recon_x, x, z_mean, z_log_var):
    reconstruction_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss,kl_loss,reconstruction_loss + kl_loss

vae = VAE().to(device)
# optimizer = optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07, weight_decay=0)
optimizer = optim.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

EPOCH = 20

############### Read the data from Excel   ####################
# Open excel
data_excel = xlrd.open_workbook('OAS_160_age.xlsx')
# get the name of sheet
names = data_excel.sheet_names()
table = data_excel.sheet_by_name(sheet_name='Sheet1')  

n_cols = table.ncols  # get the colums of sheet
n_rows = table.nrows  # get the rows of sheet

cols_data = table.col_values(0, start_rowx=0, end_rowx=None)
row_lenth = table.row_len(0)  # get the length of a row

save_path = 'ED_vae_age_OAS160.pkl'

loss_total_arr = []
loss_recon_arr = []
loss_kl_arr = []

datapath1 = '/.../'
datapath2 = '/.../'
datapath3 = '/.../'
datapath = '/.../'
for epoch in range(1, EPOCH):
    vae.train()
    total_loss = 0
    for i in range(n_rows):
        age_arr = np.ones((1, 1, 1))  .

        ###### For OAS
        name = int(table.row_values(i, start_colx=0, end_colx=1)[0])
        age = table.row_values(i, start_colx=1, end_colx=2)
        data_img = nib.load(datapath + str(name) + '.nii.gz')
        age_arr = age_arr * age

        ###### For HCPD
        # if i <= 255:
        #     name = table.row_values(i, start_colx=0, end_colx=1)
        #     age = table.row_values(i, start_colx=1, end_colx=2)
        #     data_img = nib.load(datapath3 + name[0] + '.nii.gz')
        #     age_arr = age_arr * age
        # elif 255 < i <= 651:
        #     name = table.row_values(i, start_colx=0, end_colx=1)
        #     age = table.row_values(i, start_colx=1, end_colx=2)
        #     data_img = nib.load(datapath1 + name[0] + '.nii.gz')
        #     age_arr = age_arr * age
        # else:
        #     name = table.row_values(i, start_colx=0, end_colx=1)
        #     age = table.row_values(i, start_colx=1, end_colx=2)
        #     data_img = nib.load(datapath2 + name[0] + '.nii.gz')
        #     age_arr = age_arr * age

        data = data_img.get_fdata()
        data = np.expand_dims(data, 0)
        data = np.expand_dims(data, 0)
        data = torch.from_numpy(data).float().to(device)
        if data.shape!=(1,1,144,192,160):
            data = data.permute(0, 1, 4, 3, 2)

        age = np.expand_dims(age_arr, 0)
        age = np.expand_dims(age, 0)
        age = torch.from_numpy(age).float().to(device)

        optimizer.zero_grad()
        recon_x, z_mean, z_log_var, z = vae.forward(data,age)
        recon_loss, KL_loss, loss= loss_func(recon_x, data, z_mean, z_log_var)
        loss.backward()
        total_loss = total_loss + loss.item()
        optimizer.step()
        #print('====> Epoch: {} Total loss: {:.4f}'.format(epoch, loss), ' recon_loss: {:.4f}'.format(recon_loss), ' KL_loss: {:.4f}'.format(KL_loss))

        loss_total_arr.append(loss.item())
        loss_recon_arr.append(recon_loss.item())
        loss_kl_arr.append(KL_loss.item())
        x1 = range(0, len(loss_total_arr))
        x2 = range(0, len(loss_recon_arr))
        x3 = range(0, len(loss_kl_arr))
        y1 = loss_total_arr
        y2 = loss_recon_arr
        y3 = loss_kl_arr
        plt.switch_backend('agg')
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, 'r-')
        plt.xlabel('iteration')
        plt.ylabel('loss_total')
        plt.subplot(3, 1, 2)
        plt.plot(x2, y2, 'r-')
        plt.xlabel('iteration')
        plt.ylabel('loss_recon')
        plt.subplot(3, 1, 3)
        plt.plot(x3, y3, 'r-')
        plt.xlabel('iteration')
        plt.ylabel('loss_kl')
        plt.tight_layout()

    print('====> Epoch: {} Total loss: {:.4f}'.format(epoch, total_loss))

recon_x_save = torch.squeeze(torch.squeeze(recon_x,0), 0).detach().cpu().numpy()
tem = nib.Nifti1Image(recon_x_save, data_img.affine)
nib.save(tem, '/..'+str(epoch)+'_OAS_ED.nii.gz')
torch.save(vae.state_dict(), save_path)
plt.savefig('/../' + "loss_OAS168_age_ED.jpg")
