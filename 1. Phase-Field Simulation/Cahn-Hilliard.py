
import numpy as np
import numba
import os
import timeit
import sys

# Import packages for drawing and Otus's method to label images
try:
    from matplotlib import pyplot as plt
    from skimage import filters
except:
    print('Package import error!')
    sys.exit(1)

# Directory setting
saved_path = './PF_Results'
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# Numba environment setting
os.environ['NUMBA_ENABLE_AVX'] = '1'

# Define the variables
delta = 10*10**(-9) # interface thickness
sigma = 100*10**(-3) # interfacial energy
D = 1*10**(-9) # diffusion coefficient
K = 3*sigma*delta
A = (3*sigma)/delta
M = D/(2*A)*13

# Parameters for calculation
h = 7.4*10**(-9)
delta_t = 6.5*10**(-11)

# Simulation Conditions
fv = 0.4
N = 852
beta = 0.1

# Calculate phi with njit
@numba.njit(parallel=True)
def CalcMu(phi):
    parenthesis = A * (phi*2-(phi**2)*6+(phi**3)*4) - 2 * K * _laplacian(phi)
    return parenthesis

# Laplacian function with jit(parallel)
@numba.njit(parallel=True)
def _laplacian(f):
    return (np.concatenate((f[:,1:], f[:,:1]),1) + np.concatenate((f[:,-1:], f[:,:-1]),1) + np.concatenate((f[1:,:], f[:1,:]),0) + np.concatenate((f[-1:,:], f[:-1,:]),0) - 4 * f)/(h*h)

# Set the variables for running script
runs = 2
t_start = 20
t_end = 100
t_step = 20
t_save = list(range(t_start, t_end+t_step, t_step))

start = timeit.default_timer()

# Main loop
for run in range(1, runs+1):
    print('----->Run:', run,'<-------')

    # Initialize the phi
    phi_out = []
    r = -1 + np.random.rand(N,N) * 2
    phi = np.full((N, N), fv) + r * beta

    # Calculate phi with respect to time steps
    for it in range(1, t_save[-1]+1):
        phi_old = phi
        phi = phi_old + delta_t * M * _laplacian(CalcMu(phi_old))
        if it in t_save:
            phi_out.append(phi)
            print('run',run,': image_', it,'is saved.')

    # Save images
    for idx in range(len(phi_out)):
        time_step_path = saved_path + '/t_{}'.format(t_save[idx])
        if not os.path.exists(time_step_path):
            os.makedirs(time_step_path)
        plt.imsave('{0}/CH_{2}_{1}.png'.format(time_step_path,t_save[idx],str(run).zfill(3)), phi_out[idx], cmap='gray')
        # Save images as Numpy data format for further useage
        #np.save('{0}/CH{2}_{1}.npy'.format(time_step_path,t_save[idx],str(run).zfill(3)), phi_out[idx])
        # Segmentation using Otsu's method
        val = filters.threshold_otsu(phi_out[idx])
        plt.imsave('{0}/CH_{2}_{1}_L.png'.format(time_step_path,t_save[idx],str(run).zfill(3)), phi_out[idx] > val, cmap='gray')
        print('saving images...: CH_{1}_{0}.png'.format(t_save[idx],str(run).zfill(3)))
        
end = timeit.default_timer()

print('Elapsed time: {:.3f} seconds'.format(end - start))
# To check where they saved
print('Saved path is {}'.format(saved_path))