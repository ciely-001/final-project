import torch
import numpy as np
from my_utils_radon import get_filter, filter_sinogram
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import astra

def FBP_Pinv(Mat, Spect, Measures):
    sino_approx_spectral = torch.log(torch.sum(Spect.binned_spectrum, dim=1)[:, None, None] / Measures.y)
    sino_approx_spectral[Measures.y == 0] = 0
    sino_approx_spectral = sino_approx_spectral[None]
    fsino_approx_spectral=(filter_sinogram(sino_approx_spectral))
    sino_approx_spectral_np = fsino_approx_spectral.cpu().numpy()# 假设sino_approx_spectral_np的维度是(1, 3, 120, 512)
    sino_approx_spectral_np = sino_approx_spectral_np.squeeze(0)  # 去掉批次维度，形状变为(3, 120, 512)

    sino_approx_spectral_np = sino_approx_spectral_np.transpose(1, 2, 0)  # 转置，形状变为(120, 512, 3)
    fbp = []
    for i in range(3):
        sino_channel = sino_approx_spectral_np[:, :, i]  # 选择第i个通道，形状为 (120, 512)

        # 使用Astra进行反向投影，假设radon是已创建的Radon操作
        _, fbp_channel = astra.create_backprojection(sino_channel,Measures.radon)
        fbp.append(fbp_channel)

    fbp_np = np.stack(fbp, axis=0)
    fbp_tensor = torch.tensor(fbp_np, dtype=torch.float32).cuda()
    fbp = fbp_tensor.unsqueeze(0)  # 如果需要将其扩展到四维（1, 3, img_size, img_size）

    x_mat = torch.linalg.lstsq(Mat.mass_attn_pseudo_spectral,
                               fbp[0, :].reshape(Spect.n_bin, Mat.img_size ** 2),
                               driver="gels").solution.reshape(Mat.n_mat, Mat.img_size, Mat.img_size)[None, :, :, :]

    # Computing metrics
    PSNR_fbp_pinv = np.zeros(Mat.n_mat)
    SSIM_fbp_pinv = np.zeros(Mat.n_mat)
    for k in range(Mat.n_mat):
        PSNR_fbp_pinv[k] = peak_signal_noise_ratio(Mat.x_mass_densities.detach().cpu().numpy()[0, k],
                                                   x_mat[0, k].detach().cpu().numpy(),
                                                   data_range=Mat.x_mass_densities[0, k].detach().cpu().numpy().max())
        SSIM_fbp_pinv[k] = structural_similarity(x_mat[0, k].detach().cpu().numpy(),
                                                 Mat.x_mass_densities[0, k].detach().cpu().numpy(),
                                                 data_range=Mat.x_mass_densities[0, k].detach().cpu().numpy().max(),
                                                 gradient=False)

    return x_mat, PSNR_fbp_pinv, SSIM_fbp_pinv
