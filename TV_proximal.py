import astra
import numpy as np

import matplotlib.pyplot as plt
from helpers import *
import numpy as np




l = 128
vol_geom, proj_geom, projector_id = make_astra_operator(l, n_angles=36)
phantom_id, data = astra.data2d.shepp_logan(vol_geom)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(data, proj_id)



plt.figure("Original Phantom")
plt.title("Original")
plt.imshow(data, cmap='gray')
plt.axis('off')

plt.figure("Sinogram")
plt.title("Sinogram")
plt.imshow(sinogram, cmap='gray')
plt.axis('off')
plt.show()



H, W = data.shape
n_pix = H * W
n_angles, n_det = sinogram.shape


# X and X^T
def A(theta_flat):
    img = theta_flat.reshape((H, W))
    sid, sino = astra.create_sino(img, proj_id)
    astra.data2d.delete(sid)
    return sino.ravel()

def AT(y_flat):
    sino = y_flat.reshape((n_angles, n_det))
    sid, bp = astra.create_backprojection(sino, proj_id)
    astra.data2d.delete(sid)
    return bp.ravel()


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
y = sinogram.ravel() + 0.05 * np.random.randn(n_angles * n_det)


# Estimate Lipschitz constant and choose step size
L = estimate_lipschitz_CT(num_iters=30)
tau = 0.25 / (L + 1e-12)



# TV regularization weight
lam_tv = 1  





max_iters = 2000
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


    rmse = np.sqrt(np.mean((x_new.reshape(H, W) - data)**2))
    rmse_hist.append(rmse)

    if (t % 50 == 0 or t == max_iters - 1):
        print(f"Iter {t+1}/{max_iters}: RMSE={rmse:.3e}")



np.save("TV_prox_rmse_history.npy", np.array(rmse_hist))
print("RMSE:", rmse_hist[-1])



rec_tv = x.reshape(H, W)
plt.figure("Original Phantom")
plt.imshow(data, cmap='gray')
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