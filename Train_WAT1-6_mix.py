import glob
import xlrd
import os
import random
import sys
import torch.nn as nn
from argparse import ArgumentParser
import torchvision.models as models
from ptflops import get_model_complexity_info
from torch.autograd import Variable
import numpy as np
import torch
from thop import profile
import torch.utils.data as Data
import SimpleITK as sitk
import matplotlib.pyplot as plt
import itertools
plt.switch_backend('agg')
from model_stage1_6_mix import WavletMono_unit_add_lvl1, WavletMono_unit_add_lvl2, WavletMono_unit_add_lvl3, WavletMono_unit_add_lvl4, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, antifoldloss
from WATFunctions_mix import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, median, generate_grid_unit

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
saveImgPath = '/.../'

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-5, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=40001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=40001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
parser.add_argument("--iteration_lvl4", type=int,
                    dest="iteration_lvl4", default=60001,
                    help="number of lvl4 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=1000,
                    help="Anti-fold loss: suggested range 1 to 10000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=0.1,
                    help="Gradient smooth loss: suggested range 0.01 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=1000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/.../',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number step for freezing the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3
iteration_lvl4 = opt.iteration_lvl4

def load_4D(name):
    # X = nib.load(name)
    # X = X.get_fdata()
    X = sitk.ReadImage(name)
    X = sitk.GetArrayFromImage(X)
    X = np.reshape(X, (1,) + X.shape)
    return X

def train_lvl1():
    model_name = "WD_LPBA_NCC_lvl1_"
    print("Training lvl1...")
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)

    loss_similarity1 = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss
    loss_zero = nn.MSELoss()
    zero_num = np.zeros((1, 3, 20, 23, 18), dtype=np.int)
    zero = torch.from_numpy(zero_num).type(torch.FloatTensor)

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    
    ############### Read the data from Excel   ####################
    # Open excel
    data_excel = xlrd.open_workbook('/../xx.xls')
    table = data_excel.sheet_by_name(sheet_name='Sheet1')  
    n_cols = table.ncols  # get the clumns of the sheet
    cols_data = table.col_values(0, start_rowx=0, end_rowx=None)
    row_lenth = table.row_len(0)  # get the length of a row

    #############  registration   ###############################
    grid = generate_grid(imgshape1)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '/../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm=0
    loss_si = 0
    loss_Ja = 0

    load_model = False
    if load_model is True:
        model_path = "/../Model/stage/WD_LPBA_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/../Model/loss/WD_LPBA_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]
    # epoch = 0
    step = 1
    while step <= iteration_lvl1:   
        for i in range(n_cols):
        cols_data = table.col_values(i, start_rowx=0, end_rowx=None)
        mid_img = median(cols_data)
        # if template_prior != 0:
        #     cols_data.append(template_prior)
        template_inital_img = nib.load(datapath+mid_img + '.nii.gz')
        template_inital_arr = template_inital_img.get_fdata()
        template_inital = torch.from_numpy(template_inital_arr).float()
        template = torch.unsqueeze(torch.unsqueeze(template_inital, 0), 0).to(device)
        if template.shape != (1, 1, 144, 192, 160):
            template = template.permute(0, 1, 4, 3, 2)

        for epoch_group in range(n_cols):   
            for sub_name in cols_data:
                if (sub_name != ''):
                    cols_data_num = cols_data_num + 1
                    sub_name_img = nib.load(datapath + sub_name +'.nii.gz')
                    sub_name_arr = sub_name_img.get_fdata()
                    sub = torch.from_numpy(sub_name_arr).float()
                    sub = torch.unsqueeze(torch.unsqueeze(sub, 0), 0).to(device)
           

                field1, warped_t1,fixed1,mov1 = model(template, sub)

            # 1 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC  = loss_multiNCC1

            field_norm = transform_unit_flow_to_flow_cuda(field1.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(field_norm, grid)
            # loss_zerof = loss_zero(field1, zero)
            # reg2 - use velocity
            _, _, x, y, z = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x
            loss_regulation1 = loss_smooth(field1)

            loss_regulation = loss_regulation1

            loss_fold = loss_antifold(field1)

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + 1* loss_regulation + 1*loss_fold #+ 0*loss_Jacobian
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])
            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()
            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            if (step % 10 == 0):
                total = format(float(loss_to)/float(10), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si)/float(10), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(10), '.6f')
                #loss_smo.append(smo)
                Jaco =  format((float(loss_Ja)/float(10)), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 40000 == 0):
                modelname = model_dir + '/stage/lvl1/' + model_name + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl1/' + model_name + str(step) + '.npy', lossall)

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + '/loss/lvl1/' + str(step) + "_lv1.jpg")

            step += 1

            if step > iteration_lvl1:
                break
        print("-------------------------- level 1 epoch pass-------------------------")
    print("level 1 Finish!")

def train_lvl2():
    model_name = "WD_LPBA_NCC_lvl2_"
    print("Training lvl2...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model1_path = "/../Model/stage/lvl1/WD_LPBA_NCC_lvl1_40000.pth"
    model1.load_state_dict(torch.load(model1_path))
    print("Loading weight for model_lvl1...", model1_path)

    # Freeze model_lvl1 weight
    for param in model1.parameters():
        param.requires_grad = False

    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, model1=model1).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss
    loss_zero = nn.MSELoss
    zero = np.zeros((1, 3, 40, 46, 36), dtype=np.int)
    zero = torch.from_numpy(zero).type(torch.FloatTensor)

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))[0:300]
    index_pair = list(itertools.permutations(names, 2))
    names2 = sorted(glob.glob('/..' + '*.nii'))[0:300]
    index_pair2 = list(itertools.permutations(names2, 2))
    index_pair.extend(index_pair2)
    random.shuffle(index_pair)

    grid = generate_grid(imgshape2)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '/../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    load_model = False
    if load_model is True:
        model_path = "/../Model/stage/WD_LPBA_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/../Model/loss/WD_LPBA_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl2:
        for pair in index_pair:
            LLL3_A = pair[0]
            img3_A = load_4D(LLL3_A)
            X1_LLL = torch.from_numpy(img3_A).float()

            LLL3_B = pair[1]
            img3_B = load_4D(LLL3_B)
            Y1_LLL = torch.from_numpy(img3_B).float()

            num1_pro = LLL3_A[0:39]
            num1_post = LLL3_A.split('/')[8].split('_')[0]
            LLL2_A = os.path.join(num1_pro + 'WD2/LLL2/', num1_post + '_LLL2.nii')
            img2_LLL_A = load_4D(LLL2_A)
            X2_LLL = torch.from_numpy(img2_LLL_A).float()
            HHH2_A = os.path.join(num1_pro + 'WD2/EM2/', num1_post + '_EM2.nii')
            img2_HHH_A = load_4D(HHH2_A)
            X2_HHH = torch.from_numpy(img2_HHH_A).float()

            num2_pro = LLL3_B[0:39]
            num2_post = LLL3_B.split('/')[8].split('_')[0]
            LLL2_B = os.path.join(num2_pro + 'WD2/LLL2/', num2_post + '_LLL2.nii')
            img2_LLL_B = load_4D(LLL2_B)
            Y2_LLL = torch.from_numpy(img2_LLL_B).float()
            HHH2_B = os.path.join(num2_pro + 'WD2/EM2/', num2_post + '_EM2.nii')
            img2_HHH_B = load_4D(HHH2_B)
            Y2_HHH = torch.from_numpy(img2_HHH_B).float()

            X1_LLL = torch.unsqueeze(X1_LLL,0).to(device).float()
            Y1_LLL = torch.unsqueeze(Y1_LLL,0).to(device).float()
            X2_LLL = torch.unsqueeze(X2_LLL,0).to(device).float()
            Y2_LLL = torch.unsqueeze(Y2_LLL,0).to(device).float()
            X2_HHH = torch.unsqueeze(X2_HHH,0).to(device).float()
            Y2_HHH = torch.unsqueeze(Y2_HHH,0).to(device).float()

            field1, field2,  warped_x1, warped_x2_lll, warped_x2_hhh, fixed1, fixed2_lll, fixed2_hhh, mov1, mov2\
               = model2(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH)

            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)

            #loss_multiNCC  = 0.5*loss_multiNCC1 + loss_multiNCC2
            loss_multiNCC = loss_multiNCC2
            field_norm = transform_unit_flow_to_flow_cuda(field2.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2

            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            # loss_zerof = loss_zero(field2, zero)
            #loss_regulation =  0.5*loss_regulation1 + loss_regulation2
            loss_regulation = loss_regulation2

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            #loss_fold = 0.5*loss_fold1+loss_fold2
            loss_fold = loss_fold2

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + 1 * loss_regulation + 1*loss_fold#loss_Jacobian (0.1,1,1.5,2)(0.01-10)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()
            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 40000 == 0):
                modelname = model_dir + '/stage/lvl2/' + model_name + str(step) + '.pth'
                torch.save(model2.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl2/' + model_name + str(step) + '.npy', lossall)

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + '/loss/lvl2/' + str(step) + "_lv2.jpg")

            # if step == freeze_step:
            #     model2.unfreeze_modellvl1()
            step += 1

            if step > iteration_lvl2:
                break
        print("-------------------------- level 2 epoch pass-------------------------")
    print("level 2 Finish!")

def train_lvl3():
    model_name = "WD_LPBA_NCC_lvl3_"
    print("Training lvl3...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model2_path = "/../Model/stage/lvl2/WD_LPBA_NCC_lvl2_40000.pth"
    model2.load_state_dict(torch.load(model2_path))
    print("Loading weight for model_lvl2...", model2_path)

    # Freeze model_lvl2 weight
    for param in model2.parameters():
        param.requires_grad = False

    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3, model2=model2).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)
    loss_similarity3 = NCC(win=7)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss
    loss_zero = nn.MSELoss
    zero = np.zeros((1, 3, 80, 92, 72), dtype=np.int)
    zero = torch.from_numpy(zero).type(torch.FloatTensor)

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))[0:300]
    index_pair = list(itertools.permutations(names, 2))
    names2 = sorted(glob.glob('/..' + '*.nii'))[0:300]
    index_pair2 = list(itertools.permutations(names2, 2))
    index_pair.extend(index_pair2)
    random.shuffle(index_pair)

    grid = generate_grid(imgshape3)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '/../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl3+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    load_model = False
    if load_model is True:
        model_path = "/../Model/stage/WD_LPBA_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/../Model/loss/WD_LPBA_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl3:
        for pair in index_pair:
            LLL3_A = pair[0]
            img3_A = load_4D(LLL3_A)
            X1_LLL = torch.from_numpy(img3_A).float()

            LLL3_B = pair[1]
            img3_B = load_4D(LLL3_B)
            Y1_LLL = torch.from_numpy(img3_B).float()

            num1_pro = LLL3_A[0:39]
            num1_post = LLL3_A.split('/')[8].split('_')[0]
            LLL2_A = os.path.join(num1_pro + 'WD2/LLL2/', num1_post + '_LLL2.nii')
            img2_LLL_A = load_4D(LLL2_A)
            X2_LLL = torch.from_numpy(img2_LLL_A).float()
            HHH2_A = os.path.join(num1_pro + 'WD2/EM2/', num1_post + '_EM2.nii')
            img2_HHH_A = load_4D(HHH2_A)
            X2_HHH = torch.from_numpy(img2_HHH_A).float()
            LLL3_A = os.path.join(num1_pro + 'WD1/LLL1/', num1_post + '_LLL1.nii')
            img3_LLL_A = load_4D(LLL3_A)
            X3_LLL = torch.from_numpy(img3_LLL_A).float()
            HHH3_A = os.path.join(num1_pro + 'WD1/EM1/', num1_post + '_EM1.nii')
            img3_HHH_A = load_4D(HHH3_A)
            X3_HHH = torch.from_numpy(img3_HHH_A).float()

            num2_pro = LLL3_B[0:39]
            num2_post = LLL3_B.split('/')[8].split('_')[0]
            LLL2_B = os.path.join(num2_pro + 'WD2/LLL2/', num2_post + '_LLL2.nii')
            img2_LLL_B = load_4D(LLL2_B)
            Y2_LLL = torch.from_numpy(img2_LLL_B).float()
            HHH2_B = os.path.join(num2_pro + 'WD2/EM2/', num2_post + '_EM2.nii')
            img2_HHH_B = load_4D(HHH2_B)
            Y2_HHH = torch.from_numpy(img2_HHH_B).float()
            LLL3_B = os.path.join(num1_pro + 'WD1/LLL1/', num2_post + '_LLL1.nii')
            img3_LLL_B = load_4D(LLL3_B)
            Y3_LLL = torch.from_numpy(img3_LLL_B).float()
            HHH3_B = os.path.join(num1_pro + 'WD1/EM1/', num2_post + '_EM1.nii')
            img3_HHH_B = load_4D(HHH3_B)
            Y3_HHH = torch.from_numpy(img3_HHH_B).float()

            X1_LLL = torch.unsqueeze(X1_LLL, 0).to(device).float()
            Y1_LLL = torch.unsqueeze(Y1_LLL, 0).to(device).float()
            X2_LLL = torch.unsqueeze(X2_LLL, 0).to(device).float()
            Y2_LLL = torch.unsqueeze(Y2_LLL, 0).to(device).float()
            X2_HHH = torch.unsqueeze(X2_HHH, 0).to(device).float()
            Y2_HHH = torch.unsqueeze(Y2_HHH, 0).to(device).float()
            X3_LLL = torch.unsqueeze(X3_LLL, 0).to(device).float()
            Y3_LLL = torch.unsqueeze(Y3_LLL, 0).to(device).float()
            X3_HHH = torch.unsqueeze(X3_HHH, 0).to(device).float()
            Y3_HHH = torch.unsqueeze(Y3_HHH, 0).to(device).float()

            field1, field2, field3, warped_x1, warped_x2_lll,  warped_x3_lll, warped_x2_hhh, warped_x3_hhh,fixed1, \
            fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, mov1, mov2, mov3,diff_up3, diff_Mhigh_x\
               = model3(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH)
            diff_up3 = torch.squeeze(diff_up3)
            diff_up3 = torch.squeeze(diff_up3)
            diff_Mhigh_x = torch.squeeze(diff_Mhigh_x)
            diff_Mhigh_x = torch.squeeze(diff_Mhigh_x)
            #diff_up3 = diff_up3.permute(1,2,3,0)
            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)

            #loss_multiNCC  = 0.25*loss_multiNCC1 + 0.5*loss_multiNCC2 + loss_multiNCC4
            loss_multiNCC =  loss_multiNCC4
            field_norm = transform_unit_flow_to_flow_cuda(field3.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2
            _, _, x3, y3, z3 = field3.shape
            field3[:, 0, :, :, :] = field3[:, 0, :, :, :] * z3
            field3[:, 1, :, :, :] = field3[:, 1, :, :, :] * y3
            field3[:, 2, :, :, :] = field3[:, 2, :, :, :] * x3
            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            #loss_regulation = 0.25*loss_regulation1 + 0.5*loss_regulation2 + loss_regulation3
            loss_regulation = loss_regulation3

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            #loss_fold = 0.25 * loss_fold1 + 0.5*loss_fold2 + loss_fold3
            loss_fold = loss_fold3

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + 1 * loss_regulation + 1*loss_fold
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()
            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 30000 == 0):
                modelname = model_dir + '/stage/lvl3/' + model_name + str(step) + '.pth'
                torch.save(model3.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl3/' + model_name + str(step) + '.npy', lossall)

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + '/loss/lvl3/' + str(step) + "_lv3.jpg")

            # if step == freeze_step:
            #     model3.unfreeze_modellvl2()
            step += 1

            if step > iteration_lvl3:
                break
        print("-------------------------- level 3 epoch pass-------------------------")
    print("level 3 Finish!")

def train_lvl4():
    model_name = "WD_LPBA_NCC_lvl4_"
    print("Training lvl4...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3,
                                      model2=model2).to(device)

    model3_path = "/../Model/stage/lvl3/WD_LPBA_NCC_lvl3_60000.pth"
    model3.load_state_dict(torch.load(model3_path))
    print("Loading weight for model_lvl3...", model3_path)

    # Freeze model_lvl3 weight
    for param in model3.parameters():
        param.requires_grad = False

    model4 = WavletMono_unit_add_lvl4(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3, imgshape4=imgshape4, model3=model3).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)
    loss_similarity3 = NCC(win=7)
    loss_similarity4 = NCC(win=9)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss
    loss_zero = nn.MSELoss
    zero = np.zeros((1, 3, 80, 92, 72), dtype=np.int)
    zero = torch.from_numpy(zero).type(torch.FloatTensor)

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '/*.nii'))[0:300]
    index_pair = list(itertools.permutations(names, 2))
    names2 = sorted(glob.glob('/..WD3/LLL3/' + '*.nii'))[0:300]
    index_pair2 = list(itertools.permutations(names2, 2))
    index_pair.extend(index_pair2)
    random.shuffle(index_pair)

    grid = generate_grid(imgshape4)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model4.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '/../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl4+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    load_model = False
    if load_model is True:
        model_path = "/../Model/stage/stage/WD_LPBA_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/../Model/stage/loss/WD_LPBA_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl4:
        for pair in index_pair:
            LLL3_A = pair[0]
            img3_A = load_4D(LLL3_A)
            X1_LLL = torch.from_numpy(img3_A).float()

            LLL3_B = pair[1]
            img3_B = load_4D(LLL3_B)
            Y1_LLL = torch.from_numpy(img3_B).float()

            num1_pro = LLL3_A[0:39]
            num1_post = LLL3_A.split('/')[8].split('_')[0]
            LLL2_A = os.path.join(num1_pro + 'WD2/LLL2/', num1_post + '_LLL2.nii')
            img2_LLL_A = load_4D(LLL2_A)
            X2_LLL = torch.from_numpy(img2_LLL_A).float()
            HHH2_A = os.path.join(num1_pro + 'WD2/EM2/', num1_post + '_EM2.nii')
            img2_HHH_A = load_4D(HHH2_A)
            X2_HHH = torch.from_numpy(img2_HHH_A).float()
            LLL3_A = os.path.join(num1_pro + 'WD1/LLL1/', num1_post + '_LLL1.nii')
            img3_LLL_A = load_4D(LLL3_A)
            X3_LLL = torch.from_numpy(img3_LLL_A).float()
            HHH3_A = os.path.join(num1_pro + 'WD1/EM1/', num1_post + '_EM1.nii')
            img3_HHH_A = load_4D(HHH3_A)
            X3_HHH = torch.from_numpy(img3_HHH_A).float()
            source_A = os.path.join(num1_pro + 'Source/', num1_post + '_source.nii')
            img2_source_A = load_4D(source_A)
            source1 = torch.from_numpy(img2_source_A).float()

            num2_pro = LLL3_B[0:39]
            num2_post = LLL3_B.split('/')[8].split('_')[0]
            LLL2_B = os.path.join(num2_pro + 'WD2/LLL2/', num2_post + '_LLL2.nii')
            img2_LLL_B = load_4D(LLL2_B)
            Y2_LLL = torch.from_numpy(img2_LLL_B).float()
            HHH2_B = os.path.join(num2_pro + 'WD2/EM2/', num2_post + '_EM2.nii')
            img2_HHH_B = load_4D(HHH2_B)
            Y2_HHH = torch.from_numpy(img2_HHH_B).float()
            LLL3_B = os.path.join(num1_pro + 'WD1/LLL1/', num2_post + '_LLL1.nii')
            img3_LLL_B = load_4D(LLL3_B)
            Y3_LLL = torch.from_numpy(img3_LLL_B).float()
            HHH3_B = os.path.join(num1_pro + 'WD1/EM1/', num2_post + '_EM1.nii')
            img3_HHH_B = load_4D(HHH3_B)
            Y3_HHH = torch.from_numpy(img3_HHH_B).float()
            source_B = os.path.join(num2_pro + 'Source/', num2_post + '_source.nii')
            img2_source_B = load_4D(source_B)
            source2 = torch.from_numpy(img2_source_B).float()

            X1_LLL = torch.unsqueeze(X1_LLL, 0).to(device).float()
            Y1_LLL = torch.unsqueeze(Y1_LLL, 0).to(device).float()
            X2_LLL = torch.unsqueeze(X2_LLL, 0).to(device).float()
            Y2_LLL = torch.unsqueeze(Y2_LLL, 0).to(device).float()
            X2_HHH = torch.unsqueeze(X2_HHH, 0).to(device).float()
            Y2_HHH = torch.unsqueeze(Y2_HHH, 0).to(device).float()
            X3_LLL = torch.unsqueeze(X3_LLL, 0).to(device).float()
            Y3_LLL = torch.unsqueeze(Y3_LLL, 0).to(device).float()
            X3_HHH = torch.unsqueeze(X3_HHH, 0).to(device).float()
            Y3_HHH = torch.unsqueeze(Y3_HHH, 0).to(device).float()
            source1 = torch.unsqueeze(source1, 0).to(device).float()
            source2 = torch.unsqueeze(source2, 0).to(device).float()

            field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, \
            fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3, source1, diff_up4, diff_Mhigh_x \
                = model4(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2)
            diff_up4 = torch.squeeze(diff_up4)
            diff_up4 = torch.squeeze(diff_up4)
            diff_Mhigh_x = torch.squeeze(diff_Mhigh_x)
            diff_Mhigh_x = torch.squeeze(diff_Mhigh_x)
            #diff_up4 = diff_up4.permute(1,2,3,0)
            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)
            loss_multiNCC6 = loss_similarity4(warped_source1, fixed4_source2)
            #loss_multiNCC  = 0.125*loss_multiNCC1 + 0.25*loss_multiNCC2 + 0.5*loss_multiNCC4 + loss_multiNCC6
            loss_multiNCC = loss_multiNCC6

            field_norm = transform_unit_flow_to_flow_cuda(field4.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2
            _, _, x3, y3, z3 = field3.shape
            field3[:, 0, :, :, :] = field3[:, 0, :, :, :] * z3
            field3[:, 1, :, :, :] = field3[:, 1, :, :, :] * y3
            field3[:, 2, :, :, :] = field3[:, 2, :, :, :] * x3
            _, _, x4, y4, z4 = field4.shape
            field4[:, 0, :, :, :] = field4[:, 0, :, :, :] * z4
            field4[:, 1, :, :, :] = field4[:, 1, :, :, :] * y4
            field4[:, 2, :, :, :] = field4[:, 2, :, :, :] * x4
            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            loss_regulation4 = loss_smooth(field4)
            #loss_regulation = 0.125*loss_regulation1 + 0.25*loss_regulation2 + 0.5*loss_regulation3 + loss_regulation4
            loss_regulation = loss_regulation4

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            loss_fold4 = loss_antifold(field4)
            loss_fold = loss_fold4
            #loss_fold = 0.125 * loss_fold1 + 0.25 * loss_fold2 + 0.5 * loss_fold3 + loss_fold4

            #if (step % 10000 != 0):
            del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
            torch.cuda.empty_cache()
            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            # if step <= 7000:
            #     loss = loss_multiNCC + smooth * loss_regulation + antifold * loss_fold  # loss_Jacobian
            # else:
            loss = loss_multiNCC + 1 * loss_regulation + 1 * loss_fold
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()

            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 2000 == 0):
                modelname = model_dir + '/stage/lvl4/' + model_name + str(step) + '.pth'
                torch.save(model4.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl4/' + model_name + str(step) + '.npy', lossall)
                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + '/loss/lvl4/' + str(step) + "_lv4.jpg")
                # del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
                # torch.cuda.empty_cache()

            # if step == freeze_step:
            #     model4.unfreeze_modellvl3()

            step += 1
            print(step)
            if step > iteration_lvl4:
                break
        print("-------------------------- level 4 epoch pass-------------------------")
    print("level 4 Finish!")

range_flow = 1
imgshape4 = (160, 184, 144)
imgshape3 = (80, 92, 72)
imgshape2 = (40, 46, 36)
imgshape1 = (20, 23, 18)

train_lvl1()
train_lvl2()
train_lvl3()
train_lvl4()

