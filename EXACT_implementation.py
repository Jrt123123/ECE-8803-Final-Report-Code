import numpy as np
import astra
from helpers import *
import matplotlib.pyplot as plt


#implement the steps in the extragradient algorithm
def h(t, mu, s):
    W = len(mu)
    t_pos = np.maximum(t, 0.0)

    result = np.zeros_like(t_pos)
    for j in range(W):
        result += s[j] * np.exp(-mu[j] * t_pos)
    
    return result


def F(x, y, Aop, I0, mus, weights):
    n = len(y)
    t = Aop.X(x)
    hx = h(t, mus, weights)
    fi = y - I0 * hx
    At_r = Aop.XT(fi)
    Fx = At_r /n
    return Fx

def proj_X(x):
    return np.maximum(x, 0.0)






# Calculate the stepsize according to theorem 2
def compute_step_size(n, I0, mus, s,lambda_max):
    sum_s_mu = np.sum(s * mus)
    gamma = n / (4 * I0 * lambda_max * sum_s_mu)
    return gamma


def compute_lambda_max(A_op, AT_op, img_shape, n_iters=20):
    x = np.random.randn(*img_shape)
    x /= np.linalg.norm(x)
    for _ in range(n_iters):
        Ax = A_op(x)
        ATAx = AT_op(Ax)
        x = ATAx / np.linalg.norm(ATAx)
    return np.sum(x * ATAx)



# excecute EXACT algorithm
def do_exact(y_meas, Aop, I0, mus, s, x0, x_star, max_iter, gamma):
    x = x0.copy().astype(np.float64)

    y = y_meas.ravel().astype(np.float64)

    rmse_hist = []

    for t in range(max_iter):
        Fx = F(x, y, Aop, I0, mus, s)
        x_half = proj_X(x - gamma * Fx)

        F_half = F(x_half, y, Aop, I0, mus, s)
        x_new = proj_X(x - gamma * F_half)

        x = x_new

        rmse = np.sqrt(np.mean((x_new - x_star.ravel())**2))
        rmse_hist.append(rmse)

        if (t % 50 == 0 or t == max_iter - 1):
                print(f"Iter {t+1}/{max_iter}: RMSE={rmse:.3e}")

    return x, rmse_hist



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
noise_level = 0.05  
y_meas = y + noise_level * np.random.randn(*y.shape)






# Compute step size
lambda_max = compute_lambda_max(Aop.X, Aop.XT, (l, l), n_iters=30)
n_measurements = n_angles * Aop.n_rays
gamma = 0.6*compute_step_size(n_measurements, I0, mus, s,lambda_max)





# Run EXACT
max_iter = 5000
x0 = np.zeros((l * l,), dtype=np.float64)
exact, rmse_hist = do_exact(
    y_meas=y_meas,
    Aop=Aop,
    I0=I0,
    mus=mus,
    s=s,
    max_iter=max_iter,
    gamma=gamma,
    x0=x0,
    x_star=x_star,
)
exact = exact.reshape(l, l)

print("last RMSE", rmse_hist[-1])




# Visualization
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(x_star, cmap='gray')
plt.title("Ground truth")
plt.colorbar()
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(exact, cmap='gray')
plt.title(f"EXACT reconstruction\nRMSE={rmse_hist[-1]:.4f}")
plt.colorbar()
plt.axis("off")

plt.subplot(1,3,3)
plt.plot(rmse_hist, label='RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE (log scale)')
plt.title("RMSE history")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

