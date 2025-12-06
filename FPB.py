# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------

import astra
import numpy as np
import matplotlib.pyplot as plt
from helpers import *



vol_geom = astra.create_vol_geom(128, 128)
proj_geom = astra.create_proj_geom(
    'parallel', 1.0, 128, np.linspace(0, np.pi, 36, False)
)

phantom_id, P = astra.data2d.shepp_logan(vol_geom)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)

noise_level = 0.05 
noise = noise_level * np.random.randn(*sinogram.shape)
sinogram_noisy = sinogram + noise





plt.figure("Phantom Image")
plt.title("Shepp-Logan Phantom")
plt.imshow(P, cmap='gray')
plt.axis('off')


plt.figure("Sinogram")
plt.imshow(sinogram,cmap='gray')
plt.title("Sinogram")
plt.axis('off')
plt.show()



sinogram_noisy_id = astra.data2d.create('-sino', proj_geom, sinogram_noisy)
rec_id = astra.data2d.create('-vol', vol_geom)

cfg = astra.astra_dict('FBP_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_noisy_id
cfg['option'] = {'FilterType': 'Ram-Lak'}

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)

rec = astra.data2d.get(rec_id)




plt.figure("Reconstructed Image using ASTRA FBP")
plt.imshow(rec, cmap='gray')
plt.title("Reconstructed Image using ASTRA FBP")
plt.axis('off')




plt.figure("Phantom vs Reconstruction", figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(P, cmap='gray')
plt.title("Original Phantom")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rec, cmap='gray')
plt.title("Reconstructed Image (FBP)")
plt.axis('off')

plt.tight_layout()



# Compute and print final RMSE
rmse_fbp = np.sqrt(np.mean((rec.ravel() - P.ravel())**2))
print(f"[FBP] Final RMSE: {rmse_fbp:.6f}")


plt.show()




