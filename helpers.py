import astra
import numpy as np
from scipy.ndimage import convolve



def soft_thresh_entry(w,ksi):
    if w>ksi:
        return w-ksi
    elif w<-ksi:
        return w+ksi
    else:
        return 0
    

def soft_thresh_vector(w,ksi):
    result = np.zeros_like(w)
    for i in range(len(w)):
        result[i] = soft_thresh_entry(w[i],ksi)
    return result


def soft_thresh_deriv_entry(w,ksi):
    if w>ksi:
        return 1
    elif w<-ksi:
        return 1
    else:
        return 0
    
def soft_thresh_deriv_vector(w,ksi):
    result = np.zeros_like(w)
    for i in range(len(w)):
        result[i] = soft_thresh_deriv_entry(w[i],ksi)
    return result


def do_ISTA(at,lam,y,projection_obj,theta0,num_iters,theta_star=None):
    iter = num_iters
    #ISTA
    alpha_t = at
    theta_t = theta0
    ISTA_error_hist = []
    for i in range(iter):
        z_t = y - projection_obj.X(theta_t)
        theta_t = soft_thresh_vector(theta_t + alpha_t*(projection_obj.XT(z_t)), alpha_t*lam)
        ISTA_error_hist.append(np.linalg.norm(theta_t - theta_star))

    return theta_t, ISTA_error_hist


def do_AMP(lam,y,projection_obj,theta0,num_iters,theta_star=None):
    iter = num_iters
    n = y.size                                    
    d = theta0.size                               
    
    #AMP
    theta_t = theta0
    AMP_error_hist = []
    AMP_list = [theta0]
    theta_t1 = theta0
    for i in range(iter):
        theta_t = AMP_list[i]
        sum = 0
        if i == 0:
            z_t = y - projection_obj.X(theta_t)
        else:
            term = theta_t + projection_obj.XT(z_t)
            for j in range(d):
                sum += 1/n * soft_thresh_deriv_entry(term[j],lam)
            z_t = y - projection_obj.X(theta_t) + sum*z_t

        theta_t1 = soft_thresh_vector(theta_t+projection_obj.XT(z_t), lam)
        AMP_list.append(theta_t1)
        AMP_error_hist.append(np.linalg.norm(theta_t1 - theta_star))

    return theta_t1, AMP_error_hist


class projection_obj:
    def __init__(self,sino,prjection_id=None):
        self.sino = sino
        self.n_angles, self.n_rays = sino.shape
        self.projection_id = prjection_id
        

    def Sino_to_y(self):
        return self.sino.ravel()
    
    def get_sino(self):
        return self.sino
    
    def y_to_sino(self,y):
        return y.reshape((self.n_angles,self.n_rays))
    
    def X(self,theta):
        img = theta.reshape((self.n_rays,self.n_rays))
        rid,res = astra.create_sino(img, self.projection_id)
        astra.data2d.delete(rid)
        return res.ravel()
    
    def XT(self,y):
        sino = self.y_to_sino(y)
        sid,res = astra.create_backprojection(sino, self.projection_id)
        astra.data2d.delete(sid)
        return res.ravel()

    


def make_astra_operator(l=128, n_angles=30):
    vol_geom = astra.create_vol_geom(l, l)
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, l, angles)

    projector_id = astra.create_projector('linear', proj_geom, vol_geom)
    return vol_geom, proj_geom, projector_id


def compute_Wz(z):
    dim = 2
    # z in n = H x W

    A1 = np.zeros_like(z)
    A2 = np.zeros_like(z)
    D1 = np.zeros_like(z)
    D2 = np.zeros_like(z)

    a = np.array([1,1])
    a1 = a.reshape(2, 1)
    a2 = a.reshape(1, 2)    
    d = np.array([1,-1])
    d1 = d.reshape(2, 1)
    d2 = d.reshape(1, 2)


    # A1 do convolution along rows
    A1 = convolve(z, a1, mode='wrap')
    # A2 do convolution along columns
    A2 = convolve(z, a2, mode='wrap')

    D1 = convolve(z, d1, mode='wrap')
    D2 = convolve(z, d2, mode='wrap')

    scale = 1.0 / (2 * np.sqrt(dim))
    Wz = scale * np.stack([A1, A2, D1, D2], axis=0)
    
    return Wz






def apply_Wt(v):

    H, W = v[0].shape
    dim = 2
    # scale = 1.0 / (2 * np.sqrt(dim))
    
    A1, A2, D1, D2 = v

    result = np.zeros((H, W))

    aT = np.array([1,1])
    a1T = aT.reshape(2, 1)
    a2T = aT.reshape(1, 2)  

    dT = np.array([1,-1])

    d1T = dT.reshape(2, 1)
    d2T = dT.reshape(1, 2)

    z1 = convolve(np.flipud(A1), a1T, mode='wrap')
    z1 = np.flipud(z1)
    z2 = convolve(np.fliplr(A2), a2T, mode='wrap')
    z2 = np.fliplr(z2)
    z3 = convolve(np.flipud(D1), d1T, mode='wrap')
    z3 = np.flipud(z3)
    z4 = convolve(np.fliplr(D2), d2T, mode='wrap')
    z4 = np.fliplr(z4)
    result = z1 + z2 + z3 + z4

    scale = 1.0 / (2 * np.sqrt(dim))

    return  scale * result




def anis_TV_shrink(u, lam):
    return np.sign(u) * np.maximum(np.abs(u) - lam, 0)


def S_tau_anisotropic(z, tau):
    d = 2
    lam = tau * 2 * np.sqrt(d)
    

    Wz = compute_Wz(z)
    A1, A2, D1, D2 = Wz
    
    #apply only on D
    D1_shrunk = anis_TV_shrink(D1, lam)
    D2_shrunk = anis_TV_shrink(D2, lam)
    
    T_lam = np.stack([A1, A2, D1_shrunk, D2_shrunk], axis=0)
    
    result = apply_Wt(T_lam)
    
    return result