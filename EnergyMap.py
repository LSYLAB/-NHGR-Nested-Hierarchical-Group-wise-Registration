import nibabel as nib
import torch
import numpy as np
import SimpleITK as sitk
import glob
import os
import cmath
########## Energy map ##########################
save_path = '/../WD3/EM3/'
LLH_path = '/../WD3/LLH3/'
LHL_path = '/../WD3/LHL3/'
HLL_path = '/../WD3/HLL3/'
LHH_path = '/../WD3/LHH3/'
HLH_path = '/../WD3/HLH3/'
HHL_path = '/../WD3/HHL3/'
HHH_path = '/../WD3/HHH3/'
names = sorted(glob.glob(LLH_path + '*.nii.gz'))[0:405]
for i in names:
    name = i.split('/')[7].split('_')[0]
    llh = sitk.ReadImage(LLH_path + name + '_LLH3.nii.gz')
    llh_arr = sitk.GetArrayFromImage(llh)
    lhl = sitk.ReadImage(LHL_path + name + '_LHL3.nii.gz')
    lhl_arr = sitk.GetArrayFromImage(lhl)
    hll = sitk.ReadImage(HLL_path + name + '_HLL3.nii.gz')
    hll_arr = sitk.GetArrayFromImage(hll)
    hlh = sitk.ReadImage(HLH_path + name + '_HLH3.nii.gz')
    hlh_arr = sitk.GetArrayFromImage(hlh)
    lhh = sitk.ReadImage(LHH_path + name + '_LHH3.nii.gz')
    lhh_arr = sitk.GetArrayFromImage(lhh)
    hhl = sitk.ReadImage(HHL_path + name + '_HHL3.nii.gz')
    hhl_arr = sitk.GetArrayFromImage(hhl)
    hhh = sitk.ReadImage(HHH_path + name + '_HHH3.nii.gz')
    hhh_arr = sitk.GetArrayFromImage(hhh)
    E = (llh_arr * llh_arr) + (lhl_arr * lhl_arr) + (hll_arr * hll_arr) + (hlh_arr * hlh_arr) + (lhh_arr * lhh_arr) + (
                hhl_arr * hhl_arr) + (hhh_arr * hhh_arr)
    En = E**0.5#cmath.sqrt(E)
    img = sitk.GetImageFromArray(E)
    sitk.WriteImage(img, save_path + name + '_EM3.nii.gz')
print("Finish!")
