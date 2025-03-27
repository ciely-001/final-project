import pandas
import numpy as np
import torch
#from torch_radon import Radon, RadonFanbeam
import astra


def create_bins(spectral_incident, bin_list, prop_factor, device):
    '''
    Create a binned spectrum. No cross detection.

    INPUTS :
    Spectral_incident : spectral incident intensity array.
    bin_list : list of energy intervals.
    prop_factor : scalar to multiply by in order to alter the total number of photon.

    OUTPUTS :
    n_bin : int. Number of bins.
    h_k : [n_bin, 150] ndarray.
    E_k : [n_bin] array. Mean energy of each bin.
    S_k : [n_bin] array. Sum of photon count for each bin.

    '''
    h_k = torch.zeros(len(bin_list) - 1, spectral_incident.shape[0], device=device)
    for k in range(len(bin_list) - 1):
        bin_k = torch.zeros(spectral_incident.shape[-1], device=device)
        bin_k[bin_list[k]:bin_list[k + 1]] = 1
        h_k[k, :] = spectral_incident * bin_k

    h_k = prop_factor * h_k
    n_bin = len(bin_list) - 1
    S_k = torch.sum(h_k, dim=1)
    E_k = torch.round(torch.sum(torch.arange(150, device=device) * h_k, dim=1) / S_k).long()

    return n_bin, prop_factor * h_k, E_k, S_k


def create_mass_attenuation_matrix(E_k, x_true_mat, pixel_size, device):
    '''
    Create the mass attenuation matrix for bones and soft tissues (n_mat=2).

    INPUT :
    E_k. [n_bin] array. Mean of each energy bin.
    pixel_size : cm per pixel.

    OUPUTS :
    Q.                 [150, n_mat] tensor.                    Contains all mass attenuation coefficients of the materials for all 150 energies.
    Q_speudo_spectral. [n_bin, n_mat] tensor.                  Contains mass att coeff but only for the mean energy of each bin.
    x_mass_density.    [n_mat, pixel_size, pixel_size] tensor. Material images in g.cm^(-3).
    '''

    # Loading the material mass attenuation data
    # Data frames are from SPEKTR (https://github.com/I-STAR/SPEKTR)
    df = pandas.read_csv('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/csv_files/mass_att02.csv')  # mm^-1
    rho_df = pandas.read_csv('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/csv_files/rho.csv')  # g cm^(-3)
    rho = torch.tensor([rho_df['adipose'][0], rho_df['calcification'][0], rho_df['fibroglandular'][0]])

    Q = torch.zeros(150, 3, device=device)
    Q[:, 0] = torch.tensor(df['adipose']) * 10 * pixel_size  # in pixel^(-1)
    Q[:, 1] = torch.tensor(df['calcification']) * 10 * pixel_size  # in pixel^(-1)
    Q[:, 2] = torch.tensor(df['fibroglandular']) * 10 * pixel_size  # Fibroglandular

    n_bin = len(E_k)
    Q_pseudo_spectral = torch.zeros(n_bin, 3, device=device)
    Q_pseudo_spectral[:, 0] = Q[E_k, 0]
    Q_pseudo_spectral[:, 1] = Q[E_k, 1]
    Q_pseudo_spectral[:, 2] = Q[E_k, 2]

    x_mass_density = torch.zeros_like(x_true_mat, device=device)

    x_mass_density[0, 0] = x_true_mat[0, 0]  # no unit
    x_mass_density[0, 1] = x_true_mat[0, 1]  # no unit
    x_mass_density[0, 2] = x_true_mat[0, 2]

    #     Q = torch.zeros(150,2, device=device)
    #     Q[:,0] = torch.tensor(df['Bones']       / rho_df['Bones'][0] ) * 10    / pixel_size**2       # in pix^2 g^(-1)
    #     Q[:,1] = torch.tensor(df['Soft Tissues']  / rho_df['Soft Tissues'][0] ) * 10 / pixel_size**2 # in pix^2 g^(-1)

    #     x_mass_density[0,0] = rho_df['Bones'][0]        * x_true_mat[0,0] * pixel_size**3 # in g.pix^(-3)
    #     x_mass_density[0,1] = rho_df['Soft Tissues'][0] * x_true_mat[0,1] * pixel_size**3 # in pix^2 g^(-1)
    return Q, Q_pseudo_spectral, x_mass_density, rho




class radon_par:
    def __init__(self, img_size, n_angles, det_count, max_angle, device):
        self.angles = np.linspace(0, max_angle, n_angles, endpoint=False)
        det_width = 1.2
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom('parallel', det_width, det_count, self.angles)
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        self.device = device


    def forward(self, x):
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0],x.shape[1], 120, x.shape[3]), dtype=np.float32)

        for i in range(x.shape[0]):  # 对第一个维度（batch_size）进行循环
            for j in range(x.shape[1]):
             _, Y[i, j] = astra.create_sino(x[i, j], self.proj_id)

        return torch.from_numpy(Y).to(self.device)


    def backward(self, x):
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0],x.shape[1], x.shape[3], x.shape[3]), dtype=np.float32)
        for i in range(x.shape[0]):  # 对第一个维度（batch_size）进行循环
            for j in range(x.shape[1]):
             _, Y[i, j] = astra.create_backprojection(x[i, j], self.proj_id)

        return torch.from_numpy(Y).to(self.device)
    def backward1(self, x):
        x = x.cpu().numpy()
        Y = np.empty((x.shape[0],x.shape[2], x.shape[2]), dtype=np.float32)
         # 对第一个维度（batch_size）进行循环
        for j in range(x.shape[0]):
            _, Y[j] = astra.create_backprojection(x[j], self.proj_id)

        return torch.from_numpy(Y).to(self.device)

    def forward1(self, x):
        return SinoFunction.apply(x, self.proj_id, self.device)


class SinoFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, proj_id, device):
        # 保存 proj_id 和 device 供反向传播使用
        ctx.proj_id = proj_id
        ctx.device = device

        # 将输入张量转换为 numpy 数组
        x_numpy = x.detach().cpu().numpy()

        # 创建一个空的 numpy 数组来存储结果
        Y_numpy = np.empty((x_numpy.shape[0], x_numpy.shape[1], 120, x_numpy.shape[3]), dtype=np.float32)

        # 对每个样本调用 astra.create_sino
        for i in range(x_numpy.shape[0]):
            for j in range(x_numpy.shape[1]):
                _, Y_numpy[i, j] = astra.create_sino(x_numpy[i, j], proj_id)

        # 将结果转换回 PyTorch 张量
        Y = torch.from_numpy(Y_numpy).to(device)

        # 保存必要的变量供反向传播使用
        ctx.save_for_backward(x)

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        # 获取保存的输入张量
        x, = ctx.saved_tensors
        grad_output_numpy = grad_output.detach().cpu().numpy()

        # 创建一个空的 numpy 数组来存储反投影结果
        grad_input_numpy = np.zeros_like(x.detach().cpu().numpy())

        # 对每个样本调用 astra 的反投影操作
        for i in range(grad_output_numpy.shape[0]):
            for j in range(grad_output_numpy.shape[1]):
               _,  grad_input_numpy[i, j] = astra.create_backprojection(grad_output_numpy[i, j], ctx.proj_id)

        # 将结果转换回 PyTorch 张量
        grad_input = torch.from_numpy(grad_input_numpy).to(ctx.device)

        return grad_input, None, None  # 返回输入的梯度和 proj_id、device 的梯度（None）
        # 计算梯度（这里假设 astra.create_sino 的反向传播是线性的）
        # 如果 astra.create_sino 的反向传播需要特殊处理，你需要在这里实现


def create_radon_op(img_size, n_angles, det_count, max_angle, geom, device):
    angles = np.linspace(0, max_angle, n_angles, endpoint=False)
    det_width = 1.2
    if geom == 'parallel':
        vol_geom = astra.create_vol_geom(img_size, img_size)
        proj_geom = astra.create_proj_geom(geom, det_width, det_count, angles)
        radon = astra.create_projector('cuda', proj_geom, vol_geom)


        # radon = Radon(resolution=img_size, angles=angles, det_count=det_count)

    elif geom == 'fanbeam':
        source_origin = 600
        origin_det = 600
        radon = astra.create_proj_geom('fanflat', det_width, det_count, angles, source_origin, origin_det)
         #radon = RadonFanbeam(resolution=img_size, angles=angles, det_count=det_count,
                              #det_spacing=det_width, source_distance=source_origin,
        #                      det_distance=origin_det, clip_to_circle=False)
    return radon



def forward_mat_op(x_mass_density, Q, S, background, radon, noise, device):
    '''
    Returns simulated measures following the forward model :
    Y ~ Poisson( S exp(-QRadon(X)) + background )

    INPUTS :
    x_mass_density [1, n_mat, img_size, img_size] tensor.  Material density images.
    Q              [150, n_mat] tensor.                    Mass Att matrix.
    S              [n_bin, 150] tensor.                    Binned spectrum from create_bins function.
    background     scalar.                                 Dark current.
    radon          TorchRadon                              From create_radon_op function.
    noise          Boolean.                                If True, applies poisson noise. Else, the expected mean Y_bar is returned.

    OUTPUT :
    Y or Y_bar. [n_bin, n_angles, det_count] tensor.
    '''
    batch, n_mat, img_size, no = x_mass_density.shape
    n_bin = S.shape[0]
    projections = []

    # Process each material individually
    for i in range(n_mat):
        # Extract each material's image (e.g., adipose, calcification, fibroglandular)
        material_image = x_mass_density[0, i].cpu().numpy()  # Extract 2D image for this material

        # Get projection using Astra
        _,x_proj = astra.create_sino(material_image, radon)  # [n_angles, det_count]

        # Append the reshaped projection (flattened to 1D)
        projections.append(x_proj.reshape(-1))  # Flatten [n_angles, det_count] to [n_angles * det_count]

    # Convert the list of projections into a tensor with shape [n_mat, n_angles * det_count]
    x_proj_reshaped = torch.tensor(projections, device=device)  # [n_mat, n_angles * det_count]

    # Apply the mass attenuation matrix Q
    Qx_proj = Q @ x_proj_reshaped  # [150, n_angles * det_count]
   # sino_id, x_proj = astra.create_sino(x_mass_density, radon)
    # x_proj = radon.forward(x_mass_density)  # Radon(X)
   # Qx_proj = Q @ x_proj.reshape(n_mat, x_proj.shape[-1] * x_proj.shape[-2])  # QRadon(X)
    Total_att = torch.exp(-Qx_proj)  # exp(-Qradon(X))
    Y_bar = S @ Total_att  # S exp(-Qradon(X))
    Y_bar += background  # S exp(-Qradon(X)) + background
    if noise:
        Y = torch.poisson(Y_bar)
        return Y.reshape(n_bin, x_proj.shape[-2], x_proj.shape[-1])
    else:
        return Y_bar.reshape(n_bin, x_proj.shape[-2], x_proj.shape[-1])

