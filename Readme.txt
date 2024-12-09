Step1: One low-frequency image and seven high-resolution images are obtained by 3D wavelet transform of the pre-processed data. (3Dwavlet.py)

Step2: Energy maps are obtained by seven high-resolution images. (EnergyMap.py)

Step3: Pretaining feature extraction network. (Train_VAE_age_ED.py)

Step4: Subgrouping based on persistent homology. (distribution_final.ipynb)

Step5: Group-wise registration is performed from local subgroups to global population. (Train_WAT1-6_mix.py)

