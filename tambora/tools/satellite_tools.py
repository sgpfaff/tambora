import numpy as np

def compute_bound(pos, vel, mass, self_pot, center_pos, center_vel, Rmax=10.0, max_iter=50):
    '''
    Determine which stars are bound through iterative 
    calculation.
    '''
    # f_xv = np.hstack((pos, vel))
    # f_center = np.hstack((center_pos, center_vel))
    # use  = np.sum((pos - center_pos)**2, axis=1) < Rmax**2
    # # iteratively refine the selection, retaining only bound particles (which have
    # # negative total energy in the satellite-centered frame using its own potential only)
    # prev_f_center = f_center
    # for i in range(max_iter):
    #     f_center = np.median(f_xv[use], axis=0)
    #     f_bound = self_pot + 0.5 * mass* np.sum((f_xv[:,3:6] - f_center[3:6])**2, axis=1) < 0
    #     if np.sum(f_bound)<=1 or all(f_center==prev_f_center): break
    #     use = f_bound * (np.sum((f_xv[:,0:3] - f_center[0:3])**2, axis=1) < Rmax**2)
    # return f_bound
    raise NotImplementedError("compute_bound is not yet implemented.")

# Support for calculating which particles are bound when 
# another set of particles is the host

def compute_tidal_radius(pos, vel, mass, self_pot, host_pos, host_vel, host_mass, G=1.0):
    '''
    Compute the tidal radius of a satellite given the positions and velocities of its particles and those of the host.
    '''
    raise NotImplementedError("compute_tidal_radius is not yet implemented.")