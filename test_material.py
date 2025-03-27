import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader_material import DataLoader_material
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import time
from classes import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Import of all methods
from FBP_Pinv import FBP_Pinv
from PWLS import PWLS
from ODPS import ODPS_method
from TDPS import TDPS_method
from OnestepLBFGS import OneStepLBFGS
from ProjectionDomainDPS import ProjectionDomainDPS_method

# Hyperparameters :
n_recon = 10

t_prime_list = [100, 250, 500, 750, 999]
step_odps_list = [0.1, 0.3, 0.5, 0.7, 0.9]
step_tdps_list = [0.1, 0.3, 0.5, 0.7, 0.9]

beta_onestepLBFGS_list = [0., 1e2, 1e3, 1e4, 1e5]
gamma_onestepLBFGS_list = [0., 1e2, 1e3, 1e4, 1e5]

beta_ProjDomainDPS_list = [0., 1, 1e4]
step_ProjDomainDPS_list = [0.1, 0.5, 0.9]

# Metrics storage :
PSNR_FBP = np.zeros([n_recon, 3])
SSIM_FBP = np.zeros([n_recon, 3])

PSNR_PWLS = np.zeros([n_recon, 3])
SSIM_PWLS = np.zeros([n_recon, 3])

PSNR_ODPS_gradapprox = np.zeros([n_recon, 3, len(t_prime_list), len(step_odps_list)])
SSIM_ODPS_gradapprox = np.zeros([n_recon, 3, len(t_prime_list), len(step_odps_list)])

PSNR_ODPS_autodiff = np.zeros([n_recon, 3, len(t_prime_list), len(step_odps_list)])
SSIM_ODPS_autodiff = np.zeros([n_recon, 3, len(t_prime_list), len(step_odps_list)])

#PSNR_TDPS = np.zeros([n_recon, 3, len(t_prime_list), len(step_tdps_list)])
#SSIM_TDPS = np.zeros([n_recon, 3, len(t_prime_list), len(step_tdps_list)])

#PSNR_OneStepLBFGS = np.zeros([n_recon, 3, len(beta_onestepLBFGS_list), len(gamma_onestepLBFGS_list)])
#SSIM_OneStepLBFGS = np.zeros([n_recon, 3, len(beta_onestepLBFGS_list), len(gamma_onestepLBFGS_list)])

PSNR_ProjDomainDPS = np.zeros(
    [n_recon, 3, len(t_prime_list), len(beta_ProjDomainDPS_list), len(step_ProjDomainDPS_list)])
SSIM_ProjDomainDPS = np.zeros(
    [n_recon, 3, len(t_prime_list), len(beta_ProjDomainDPS_list), len(step_ProjDomainDPS_list)])

my_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomApply([transforms.RandomRotation((180, 180))], p=1)]
)

# Val dataset
data_val = DataLoader_material(img_dir='C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yhvalidation_data/',
                               material_list=['Phantom_Adipose', 'Phantom_Calcification', 'Phantom_Fibroglandular'],
                                transform=my_transforms)
Data_loader_val = DataLoader(data_val, batch_size=1, shuffle=True)

for n_rec in range(n_recon):
    # Load next reference images
    x_true_mat, index = next(iter(Data_loader_val))
    x_true_mat = x_true_mat.float().to(device)
    x_true_mat_np = x_true_mat.detach().cpu().numpy()

    # Create measures
    Spect = SpectrumClass(bin_list=[10,40,60,120],
                          prop_factor=0.2,
                          device=device)

    Mat = MaterialClass(x_data=x_true_mat,
                        pixel_size=0.1,
                        Spect=Spect,
                        device=device)

    Measures = MeasureClass(Mat,
                            Spect,
                            sino_shape=[120, 512],
                            max_angle=np.pi,
                            geom='parallel',
                            background=0,
                            device=device)

    ODPS = ODPSClass()
    TDPS = TDPSClass()

    # FBP + Pinv
    x_scout, PSNR_fbp_pinv, SSIM_fbp_pinv = FBP_Pinv(Mat, Spect, Measures)
    PSNR_FBP[n_rec] = PSNR_fbp_pinv
    SSIM_FBP[n_rec] = SSIM_fbp_pinv
 #   fig, ax = plt.subplots(1, Mat.n_mat, figsize=(15, 15))
 #   for k in range(3):
  #      ax[k].imshow(x_scout.detach().cpu()[0, k], cmap='gray')
  #      ax[k].axis('off')
  #  plt.show()

    # PWLS + Pinv
    x_mat_pwls, x_spec_pwls, PSNR_pwls, SSIM_pwls = PWLS(Mat, Spect, Measures,
                                                         x_scout,
                                                         n_iter=200,
                                                         delta=1e-3,
                                                         beta_prior=torch.tensor([1e10,3e10,5e10], device=device),
                                                         device=device)
    PSNR_PWLS[n_rec] = PSNR_pwls[-1]
    SSIM_PWLS[n_rec] = SSIM_pwls[-1]
 #   x_pwls_np = x_mat_pwls.detach().cpu().numpy()
 #   fig, ax = plt.subplots(1, Mat.n_mat, figsize=(15, 15))
 #   for k in range(Mat.n_mat):
 #       ax[k].imshow(x_pwls_np[0, k], cmap='gray')
 #       ax[k].axis('off')
 #   plt.show()
 #   fig, ax = plt.subplots(1, Spect.n_bin, figsize=(20, 15))
#    for k in range(Spect.n_bin):
 #       ax[k].imshow(x_spec_pwls[0, k].detach().cpu().numpy(), cmap='gray')
 #       ax[k].axis('off')
 #   plt.show()

    for t_prime in range(len(t_prime_list)):

        for step in range(len(step_odps_list)):
            # ODPS gradapprox
            grad_approx = True
          #  start = time.time()
            x_odps_gradapprox, PSNR_odps_gradapprox, SSIM_odps_gradapprox = ODPS_method(Mat, Spect, Measures, ODPS,
                                                                                        x_scout=x_scout,
                                                                                        t_prime=t_prime_list[t_prime],
                                                                                        step=torch.tensor(
                                                                                            [step_odps_list[step],
                                                                                       step_odps_list[step],step_odps_list[step]],
                                                                                            device=device)[None, :,
                                                                                             None, None],
                                                                                        grad_approx=grad_approx,
                                                                                        device=device)
            PSNR_ODPS_gradapprox[n_rec, :, t_prime, step] = PSNR_odps_gradapprox[-1, :]
            SSIM_ODPS_gradapprox[n_rec, :, t_prime, step] = SSIM_odps_gradapprox[-1, :]
         #   end = time.time()
         #   print(f'time : {end - start}')
         #   fig, ax = plt.subplots(1, 3, figsize=(15, 15))
         #   img1 = x_odps_gradapprox[0, 0].detach().cpu().numpy()
         #   img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))  # 归一化到[0, 1]
         #   ax[0].imshow(img1, cmap='gray')
         #   ax[0].axis('off')
         #   img2 = x_odps_gradapprox[0, 1].detach().cpu().numpy()
         #   img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
         #   ax[1].imshow(img2, cmap='gray')
         #   ax[1].axis('off')
          #  ax[2].imshow(x_odps_gradapprox[0, 2].detach().cpu().numpy(), cmap='gray')
          #  ax[2].axis('off')
          #  plt.show()
            # ODPS autodiff
        #    grad_approx = False
           # start = time.time()
        #    x_odps_autodiff, PSNR_odps_autodiff, SSIM_odps_autodiff = ODPS_method(Mat, Spect, Measures, ODPS,
        #                                                                          x_scout=x_scout,
        #                                                                          t_prime=t_prime_list[t_prime],
        #                                                                          step=torch.tensor(
        #                                                                              [step_odps_list[step],
         #                                                                              step_odps_list[step],step_odps_list[step]],
         #                                                                             device=device)[None, :, None,
         #                                                                              None],
         #                                                                         grad_approx=grad_approx,
         #                                                                         device=device)
         #   PSNR_ODPS_autodiff[n_rec, :, t_prime, step] = PSNR_odps_autodiff[-1, :]
         #   SSIM_ODPS_autodiff[n_rec, :, t_prime, step] = SSIM_odps_autodiff[-1, :]
            #end = time.time()
           # print(f'time : {end - start}')
           # fig, ax = plt.subplots(1, 3, figsize=(15, 15))
           # ax[0].imshow(x_odps_autodiff[0, 0].detach().cpu().numpy(), cmap='gray')
           # ax[0].axis('off')
           # ax[1].imshow(x_odps_autodiff[0, 1].detach().cpu().numpy(), cmap='gray')
           # ax[1].axis('off')
           # ax[2].imshow(x_odps_autodiff[0, 2].detach().cpu().numpy(), cmap='gray')
           # ax[2].axis('off')
           # plt.show()
    #    for step in range(len(step_tdps_list)):
            # TDPS

    #        x_tdps, x_tdps_pseudo_spectral, PSNR_tdps, SSIM_tdps = TDPS_method(Mat, Spect, Measures, TDPS,
     #                                                                          x_scout=x_scout,
     #                                                                          t_prime=t_prime_list[t_prime],
     #                                                                          step=torch.tensor([step_tdps_list[step],
      #                                                                                            step_tdps_list[step]],
       #                                                                                          device=device)[None, :,
       #                                                                             None, None],
       #                                                                        device=device)
       #     PSNR_TDPS[n_rec, :, t_prime, step] = PSNR_tdps[-1, :]
       #     SSIM_TDPS[n_rec, :, t_prime, step] = SSIM_tdps[-1, :]

    #for beta in range(len(beta_onestepLBFGS_list)):
       # for gamma in range(len(gamma_onestepLBFGS_list)):
          #  x_lbfgs, PSNR, SSIM = OneStepLBFGS(Mat, Spect, Measures,
           #                                    x_scout,
          #                                     beta_onestepLBFGS_list[beta],
          #                                     gamma_onestepLBFGS_list[gamma],
          #                                     device)
          #  PSNR_OneStepLBFGS[n_rec, :, beta, gamma] = PSNR
          #  SSIM_OneStepLBFGS[n_rec, :, beta, gamma] = SSIM
    for t_prime in range(len(t_prime_list)):
        for beta in range(len(beta_ProjDomainDPS_list)):
            for step in range(len(step_ProjDomainDPS_list)):
                x_ProjDomainDPS, PSNR, SSIM = ProjectionDomainDPS_method(Mat, Spect, Measures, ODPS,
                                                                         x_scout,
                                                                        beta=beta_onestepLBFGS_list[beta],
                                                                         t_prime=t_prime_list[t_prime],
                                                                         step=torch.tensor(
                                                                             [step_ProjDomainDPS_list[step],
                                                                              step_ProjDomainDPS_list[step],step_ProjDomainDPS_list[step]],
                                                                             device=device)[None, :, None, None],
                                                                         device=device)
                PSNR_ProjDomainDPS[n_rec, :, t_prime, beta, step] = PSNR[-1]
                SSIM_ProjDomainDPS[n_rec, :, t_prime, beta, step] = SSIM[-1]
                fig, ax = plt.subplots(1, Mat.n_mat, figsize=(15, 15))
                for k in range(Mat.n_mat):
                    ax[k].imshow(x_ProjDomainDPS[0, k].detach().cpu(), cmap='gray')
                    ax[k].axis('off')
                plt.show()
# Saves :

np.save('HyperparametersSearch/t_prime_list.npy', np.asarray(t_prime_list))
np.save('HyperparametersSearch/step_odps_list.npy', np.asarray(step_odps_list))
#np.save('HyperparametersSearch/step_tdps_list.npy', np.asarray(step_tdps_list))

#np.save('HyperparametersSearch/beta_onestepLBFGS_list.npy', np.asarray(beta_onestepLBFGS_list))
#np.save('HyperparametersSearch/gamma_onestepLBFGS_list.npy', np.asarray(gamma_onestepLBFGS_list))

#np.save('HyperparametersSearch/beta_ProjDomainDPS_list.npy', np.asarray(beta_ProjDomainDPS_list))
#np.save('HyperparametersSearch/step_ProjDomainDPS_list.npy', np.asarray(step_ProjDomainDPS_list))

np.save('HyperparametersSearch/PSNR_FBP', np.asarray(PSNR_FBP))
np.save('HyperparametersSearch/SSIM_FBP', np.asarray(SSIM_FBP))

np.save('HyperparametersSearch/PSNR_PWLS', np.asarray(PSNR_PWLS))
np.save('HyperparametersSearch/SSIM_PWLS', np.asarray(SSIM_PWLS))

np.save('HyperparametersSearch/PSNR_ODPS_gradapprox', np.asarray(PSNR_ODPS_gradapprox))
np.save('HyperparametersSearch/SSIM_ODPS_gradapprox', np.asarray(SSIM_ODPS_gradapprox))

np.save('HyperparametersSearch/PSNR_ODPS_autodiff', np.asarray(PSNR_ODPS_autodiff))
np.save('HyperparametersSearch/SSIM_ODPS_autodiff', np.asarray(SSIM_ODPS_autodiff))

#np.save('HyperparametersSearch/PSNR_TDPS', np.asarray(PSNR_TDPS))
#np.save('HyperparametersSearch/SSIM_TDPS', np.asarray(SSIM_TDPS))

#np.save('HyperparametersSearch/PSNR_OneStepLBFGS', np.asarray(PSNR_OneStepLBFGS))
#np.save('HyperparametersSearch/SSIM_OneStepLBFGS', np.asarray(SSIM_OneStepLBFGS))

np.save('HyperparametersSearch/PSNR_ProjDomainDPS', np.asarray(PSNR_ProjDomainDPS))
np.save('HyperparametersSearch/SSIM_ProjDomainDPS', np.asarray(SSIM_ProjDomainDPS))