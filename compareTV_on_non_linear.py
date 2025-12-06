import numpy as np
import astra
from helpers import *
import matplotlib.pyplot as plt


def h(t, mu, s):
    W = len(mu)
    t_pos = np.maximum(t, 0.0)

    result = np.zeros_like(t_pos)
    for j in range(W):
        result += s[j] * np.exp(-mu[j] * t_pos)
    
    return result



# run on small image
l = 32
n_angles = 180  


#astra projection operator
vol_geom, proj_geom, projector_id = make_astra_operator(l, n_angles=n_angles)
phantom_id, x_star = astra.data2d.shepp_logan(vol_geom)
sinogram_id_lin, line_integrals = astra.create_sino(x_star, projector_id)
astra.data2d.delete(phantom_id)
astra.data2d.delete(sinogram_id_lin)
Aop = projection_obj(line_integrals, projector_id)



# frequncy spectrums
mus = np.array([0.05, 0.10, 0.20, 0.50, 0.90], dtype=np.float32)
s = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float32)




I0 = 1e5
y = I0 * h(line_integrals, mus, s)
noise_level = 0.005  
y_meas = y + noise_level * np.random.randn(*y.shape)


# Reconstruct the effective sinogram for the linearized problem
I0_eff = I0
eps = 0.001  # to avoid log(0)
sinogram = -np.log((y_meas + eps) / I0_eff)
sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)


Aop = projection_obj(line_integrals, projector_id)





H = W = l
n_pix = H * W
n_angles, n_det = sinogram.shape

# Linear forward and adjoint using the same ASTRA operator
def A(theta_flat):
    """
    Forward projection: image -> sinogram (flattened)
    """
    return Aop.X(theta_flat)  # already returns (n_angles * n_det,)

def AT(y_flat):
    """
    Backprojection: sinogram -> image (flattened)
    """
    return Aop.XT(y_flat)


def estimate_lipschitz_CT(num_iters=30):
    v = np.random.randn(n_pix)
    v /= np.linalg.norm(v) + 1e-12
    norm = 0.0
    for _ in range(num_iters):
        Av = A(v)
        ATAv = AT(Av)
        norm = np.linalg.norm(ATAv)
        if norm < 1e-12:
            break
        v = ATAv / (norm + 1e-12)
    return norm




# noisy measurements
y = sinogram.ravel() # + 0.05 * np.random.randn(n_angles * n_det)


# Estimate Lipschitz constant and choose step size
L = estimate_lipschitz_CT(num_iters=30)
tau = 0.25 / (L + 1e-12)



# TV regularization weight
lam_tv = 0.5  



max_iters = 8000
x = np.zeros(n_pix)         
rmse_hist = []

for t in range(max_iters):
    # calculate z = x - eta*grad
    Ax = A(x)
    grad = AT(Ax - y)
    z = x - tau * grad

    #prox update
    z_img = z.reshape(H, W)
    x_img_new = S_tau_anisotropic(z_img, tau)
    x_new = x_img_new.ravel()

    x = x_new


    rmse = np.sqrt(np.mean((x_new.reshape(H, W) - x_star)**2))
    rmse_hist.append(rmse)

    if (t % 50 == 0 or t == max_iters - 1):
        print(f"Iter {t+1}/{max_iters}: RMSE={rmse:.3e}")



np.save("TV_prox_rmse_history.npy", np.array(rmse_hist))
print("RMSE:", rmse_hist[-1])



rec_tv = x.reshape(H, W)
plt.figure("Original Phantom")
plt.imshow(x_star, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.figure("TV Proximal Gradient Reconstruction")
plt.imshow(rec_tv, cmap='gray')
plt.title("TV-regularized Reconstruction (approx prox)")
plt.axis('off')


plt.figure("RMSE History")
# plt.semilogy(rmse_hist)
plt.plot(rmse_hist)
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE vs. Iteration (TV-PGD)")
plt.grid(True, which='both', ls='--', alpha=0.4)

plt.show()