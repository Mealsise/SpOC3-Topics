# Copyright 2024 European Space Agency
#
# This file is shipped via the optimize.esa.int web framework.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. Obtain one at http://mozilla.org/MPL/2.0/.
#
# Usage example:
# x = numpy.random.uniform(-1, 1, 22)
# score = udp.fitness(x)

import heyoka as hy
import pygmo as pg
import numpy as np
import scipy
import PIL
import time

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm


def propagate_formation(dx0, stm):
    """From some initial (relative) position and velocities returns new (relative) positions at
    some future time (defined by the stm).
    Args:
        dx0 (`np.array` (N, 6)): initial relative positions and velocities.
        stm (`np.array` (6,6)): the state transition matrix at some future time.
    Returns:
        np.array (N,3): propagated positions
    """
    dxT = stm @ dx0.T
    # We return only the positions
    return dxT.T[:, :3]


def stm_factory(ic, T, mu, M, verbose=True):
    """Constructs all the STMS and reference trajectory in a CR3BP dynamics
    Args:
        ic (`np.array` (N, 6)): initial conditions (absolute).
        T (`float`): propagation time
        mu (`float`): gravity parameter
        M (`int`): number of grid points (observations)
        verbose (boolean): print time it took to build Taylor integrator and STMs
    Returns:
        (ref_state (M, 6), stms (M,6,6)): the propagated state and stms
    """
    # ----- We assemble the CR3BP equation of motion --------
    # The state
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    xarr = np.array([x, y, z, vx, vy, vz])
    # The dynamics
    r_1 = hy.sqrt((x + hy.par[0]) ** 2 + y**2 + z**2)
    r_2 = hy.sqrt((x - (1 - hy.par[0])) ** 2 + y**2 + z**2)
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = (
        2 * vy
        + x
        - (1 - hy.par[0]) * (x + hy.par[0]) / (r_1**3)
        - hy.par[0] * (x + hy.par[0] - 1) / (r_2**3)
    )
    dvydt = -2 * vx + y - (1 - hy.par[0]) * y / (r_1**3) - hy.par[0] * y / (r_2**3)
    dvzdt = -(1 - hy.par[0]) / (r_1**3) * z - hy.par[0] / (r_2**3) * z
    # This array contains the expressions (r.h.s.) of our dynamics
    farr = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    # We now compute the variational equations
    # 1 - Define the symbols
    symbols_phi = []
    for i in range(6):
        for j in range(6):
            # Here we define the symbol for the variations
            symbols_phi.append("phi_" + str(i) + str(j))
    phi = np.array(hy.make_vars(*symbols_phi)).reshape((6, 6))

    # 2 - Compute the gradient
    dfdx = []
    for i in range(6):
        for j in range(6):
            dfdx.append(hy.diff(farr[i], xarr[j]))
    dfdx = np.array(dfdx).reshape((6, 6))

    # 3 - Assemble the expressions for the r.h.s. of the variational equations
    dphidt = dfdx @ phi

    dyn = []
    for state, rhs in zip(xarr, farr):
        dyn.append((state, rhs))
    for state, rhs in zip(phi.reshape((36,)), dphidt.reshape((36,))):
        dyn.append((state, rhs))

    # These are the initial conditions on the variational equations (the identity matrix)
    ic_var = np.eye(6).reshape((36,)).tolist()

    start_time = time.time()
    ta = hy.taylor_adaptive(
        # The ODEs.
        dyn,
        # The initial conditions (do not matter, they will change)
        [0.1] * 6 + ic_var,
        # Operate below machine precision
        # and in high-accuracy mode.
        tol=1e-16,
    )
    if verbose:
        print(
            "--- %s seconds --- to build the Taylor integrator -- (do this only once)"
            % (time.time() - start_time)
        )
    # We set the Taylor integration param
    ta.pars[:] = [mu]
    # We set the ic
    ta.state[:6] = ic
    ta.state[6:] = ic_var
    ta.time = 0.0
    # The time grid
    t_grid = np.linspace(0, T, M)
    # We integrate
    start_time = time.time()
    sol = ta.propagate_grid(t_grid)
    if verbose:
        print("--- %s seconds --- to construct all stms" % (time.time() - start_time))

    ref_state = sol[4][:, :6]
    stms = sol[4][:, 6:].reshape(M, 6, 6)
    return (ref_state, stms)


class orbital_golomb_array:
    def __init__(
        self,
        n_sat: int,
        ic: list,
        T: float,
        n_meas=3,
        mu=0.01215058560962404,
        scaling_factor=1e-4,
        inflation_factor=1.5,
        grid_size=20,
        verbose = True
    ):
        """Constructs a UDP (User Defined Problem) compatible with pagmo/pygmo and representing the design
        of a ballistic formation flight around a nominal CR3BP trajectory, able to perform a good interferometric
        reconstruction of the image contained in *img_path*.
        Args:
            n_sat (`int`): Number of satellites in the formation.
            ic (`list`): Initial conditions of the reference CR3BP solution.
            T (`float`): Time of flight for the measurments
            mu (`float`): parameter of the CR3BP
            n_meas (`int`): Number of interferometric measurments performed along the trajectory
                           (assumed at equally spaced time intervals).
            scaling_factor (`float`, optional): The initial positions and velocities will be scaled down by this factor.
                             Defaults to 1e-4.
            inflation_factor (`float`, optional): The allowed formation inflation. (outside this radius satellites are no longer considered)
            grid_size (int, optional): Size of the Golomb grid.
            verbose (boolean): print time it took to build Taylor integrator and STMs
        """
        # Init data members
        self.n_sat = n_sat
        self.ic = ic
        self.T = T
        self.n_meas = n_meas
        self.scaling_factor = scaling_factor
        self.grid_size = grid_size
        self.mu = mu
        self.inflation_factor = inflation_factor
        self.verbose = verbose

        # We construct the various STMs and reference trajectory
        self.ref_state, self.stms = stm_factory(ic, T, mu, n_meas, self.verbose)

        self.use_custom_bounds = False
        self.custom_lower_bounds = [-1.0] * self.n_sat * 3 + [-10.0] * self.n_sat * 3
        self.custom_upper_bounds = [1.0] * self.n_sat * 3 + [10.0] * self.n_sat * 3

    def set_custom_bounds_switch(self, tight: bool):
        """
        Enable or disable the use of tighter bounds.
        
        Args:
            tight (bool): If True, use the tighter bounds. Otherwise, use default bounds.
        """
        self.use_custom_bounds = tight

    def set_custom_bounds(self, lower_bounds: list[float], upper_bounds: list[float]):
        """
        Set custom bounds for optimization.
        
        Args:
            lower_bounds (list[float]): Custom lower bounds.
            upper_bounds (list[float]): Custom upper bounds.
        """
        self.custom_lower_bounds = []
        self.custom_upper_bounds = []
        for component in range(6):
            self.custom_lower_bounds.extend([lower_bounds[component]] * self.n_sat)
            self.custom_upper_bounds.extend([upper_bounds[component]] * self.n_sat)



    def get_bounds(self):
        """Return bounds for optimization. If satellites are locked, their bounds are constrained to their current values."""
        if self.use_custom_bounds:
            return self.custom_lower_bounds, self.custom_upper_bounds

        lower_bounds = [-1.0] * self.n_sat * 3 + [-10.0] * self.n_sat * 3  # x, y, z lower bounds, vx, vy, vz lower bounds
        upper_bounds = [1.0] * self.n_sat * 3 + [10.0] * self.n_sat * 3  # x, y, z upper bounds, vx, vy, vz upper bounds

        return lower_bounds, upper_bounds


    def get_nix(self):
        """
        Get number of integer variables in the chromosome/decision vector.

        Returns:
            int: number of integer variables.
        """
        # the chromosome exists solely of float variables.
        return 0

    # Mandatory method in the UDP pygmo interface
    # (returns the fitness of the chromosome [obj1, obj2 ..., ec1, ec2, ...,iec1, iec2...]
    def fitness(self, x):
        return self.fitness_impl(x)

    # Plots the representation of the chromosome in several graphs
    def plot(self, x, figsize=(15, 4)):
        return self.fitness_impl(x, plotting=True, figsize=figsize)

    def plot_simulated_reconstruction(
        self, x, M=100, grid_size=256, image_path="data/nebula.jpg"
    ):
        """_summary_

        Args:
            x (`list` of length N): Chromosome contains initial relative positions and velocities of each satellite:
            Example: x = [ dx0_N1, dx0_N2, ..., dx0_NN, dy0_N1, dy0_N2, ..., dy0_NN , ...... , dvz0_N1, dvz0_N2, ..., dvz0_NN]
            M (`int`): Number of interferometric measurments performed along the trajectory
                           (assumed at equally spaced time intervals).
            grid_size (int, optional): _description_. Defaults to 256.
            image_path (str, optional): _description_. Defaults to "data/nebula.jpg".
        """        

        #  Time of flight for the measurments
        T = self.T

        _, stms = stm_factory(self.ic, T, self.mu, M, self.verbose)

        # 1) Decode the chromosomes into (x, y, z, vx, vy, vz) of the satellites.
        N = self.n_sat

        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[      : N], 
                                                              x[N     : 2 * N], 
                                                              x[2 * N : 3 * N],
                                                              x[3 * N : 4 * N],
                                                              x[4 * N : 5 * N],
                                                              x[5 * N : ],
                                                              )]
        )


        # We now propagate all these relative positions to the measurment points. We do this accounting for the formation size
        rel_pos = []
        for stm in stms:
            # We scale the initial positions and velocities
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            # We store the relative positions in the original 'units'
            rel_pos.append(fc / self.scaling_factor)
        rel_pos = np.array(rel_pos)

        # For each observation point we construct the corresponding Golomb Array
        gs_xy = []  # This will contain all the Golomb Arrays at each observation point
        g_xy = np.zeros(
            (grid_size, grid_size)
        )  # This will contain all the positions cumulatively (for plotting)

        # For each observation point we construct the corresponding Golomb Array
        gs_xz = []  # This will contain all the Golomb Arrays at each observation point
        g_xz = np.zeros(
            (grid_size, grid_size)
        )  # This will contain all the positions cumulatively (for plotting)

        # For each observation point we construct the corresponding Golomb Array
        gs_yz = []  # This will contain all the Golomb Arrays at each observation point
        g_yz = np.zeros(
            (grid_size, grid_size)
        )  # This will contain all the positions cumulatively (for plotting)

        for k in range(M):

            gs_xy.append(np.zeros((grid_size, grid_size)))
            gs_xz.append(np.zeros((grid_size, grid_size)))
            gs_yz.append(np.zeros((grid_size, grid_size)))

            points_3D = rel_pos[k]                
            # Account for an added factor allowing the formation to spread.
            points_3D = points_3D / (self.inflation_factor) 

            # and removing the points outside [-1,1] (cropping wavelengths here)
            points_3D = points_3D[np.max(points_3D, axis=1) < 1 ]
            points_3D = points_3D[np.min(points_3D, axis=1) > -1]

            # Interpret now the 3D positions [-1,1] as points on a grid.
            pos3D = (points_3D * grid_size / 2).astype(int)
            pos3D = pos3D + int(grid_size / 2)
            

            for i, j, k_ in pos3D:
                gs_xy[k][i, j] = 1
                g_xy[i, j] = 1

                gs_xz[k][i, k_] = 1
                g_xz[i, k_] = 1

                gs_yz[k][j, k_] = 1
                g_yz[j, k_] = 1

        def plot_recon(gs, g):
            # We Simulate the interferometric measurement
            otf = np.zeros((grid_size * 2 - 1, grid_size * 2 - 1))
            for one_g in gs:
                tmp = scipy.signal.correlate(one_g, one_g, mode="full")
                otf = otf + tmp
            otf[abs(otf) < 0.1] = 0
            otf[abs(otf) > 1] = 1
            otf = np.fft.fftshift(otf)

            I_o = PIL.Image.open(image_path)
            I_o = np.asarray(I_o.resize((511, 511)))
            imo_fft = np.fft.fft2(I_o)
            imr_fft = imo_fft * otf  # Hadamard product here
            I_r = abs(np.fft.ifft2(imr_fft))

            # We plot
            fig = plt.figure(figsize=(15, 3))
            ax = fig.subplots(1, 4)
            ax[0].imshow(I_o, cmap="gray")
            ax[0].axis("off")
            ax[0].set_title("Image")
            ax[1].imshow(I_r, cmap="gray")
            ax[1].axis("off")
            ax[1].set_title("Reconstruction")
            ax[2].imshow(g, cmap="gray")
            ax[2].axis("off")
            ax[2].set_title("Golomb Array Traj")
            ax[3].imshow(otf, cmap="gray")
            ax[3].axis("off")
            ax[3].set_title("Optical Transfer Function")
            plt.show()

        print('XY')
        plot_recon(gs_xy, g_xy)
        print('XZ')
        plot_recon(gs_xz, g_xz)
        print('YZ')
        plot_recon(gs_yz, g_yz)


    # Here is where the action takes place
    def fitness_impl(self, x, plotting=False, figsize=(15, 10)):
        """ Fitness function

        Args:
            x (`list` of length N): Chromosome contains initial relative positions and velocities of each satellite:
            Example: x = [ dx0_N1, dx0_N2, ..., dx0_NN, dy0_N1, dy0_N2, ..., dy0_NN , ...... , dvz0_N1, dvz0_N2, ..., dvz0_NN]
            plotting (bool, optional): Plot satellites on grid at each measurement and corresponding auto-correlation function and fill factors. Defaults to False.
            figsize (tuple, optional): Figure size. Defaults to (15, 10).
        Returns:
            float: fitness of corresponding chromosome x.
        """        
        # 1) Decode the chromosomes into (x, y, z, vx, vy, vz) of the satellites.
        N = self.n_sat

        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[      : N], 
                                                              x[N     : 2 * N], 
                                                              x[2 * N : 3 * N],
                                                              x[3 * N : 4 * N],
                                                              x[4 * N : 5 * N],
                                                              x[5 * N : ],
                                                              )]
        )

        # 2) We now propagate all these relative positions to the measurment points. We do this accounting for the formation size
        rel_pos = []
        for stm in self.stms:
            # We scale the initial positions and velocities
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            # We store the relative positions in the original 'units'
            rel_pos.append(fc / self.scaling_factor)
        rel_pos = np.array(rel_pos)

        # 3) At each observation epoch we compute the fill factor
        # See:
        # Memarsadeghi, Nargess, Ryan D. Joseph, John C. Kaufmann, and Byung Suk Lee.
        # "Golomb Patterns, Astrophysics, and Citizen Science Games." IEEE Access 10 (2022): 76125-76135.

        fill_factor = []

        if plotting:
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(2, self.n_meas * 3, hspace=0.02, wspace=0.02)
            # fig.suptitle('Placement of satellites (red squares) with respect to mothership (M)\nAutocorrelation matrix and corresponding fill factors.', fontsize=16, fontweight='bold', y=1.05)
            plt.axis("off")
            axs = gs.subplots(sharex=False, sharey=False)
            axs = axs.ravel()

        for k in range(self.n_meas):

            points_3D = rel_pos[k]   
            # Account for an added factor allowing the formation to spread. (Except for first observation, k == 0)
            if k != 0:
                points_3D = points_3D / (self.inflation_factor) 

            # and removing the points outside [-1,1] (cropping wavelengths here)
            points_3D = points_3D[np.max(points_3D, axis=1) < 1 ]
            points_3D = points_3D[np.min(points_3D, axis=1) > -1]

            # Interpret now the 3D positions [-1,1] as points on a grid.
            pos3D = (points_3D * self.grid_size / 2)
            pos3D = pos3D + int(self.grid_size / 2)
            pos3D = pos3D.astype(int)

            # Compute uv plane fill factor
            # Here we use the (https://stackoverflow.com/questions/62026550/why-does-scipy-signal-correlate2d-use-such-an-inefficient-algorithm)
            # correlate and ot correlate2d. We must then consider a threshold to define zero as the frequency domain inversion has numerical roundoffs

            I = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            for i, j, k_ in pos3D:
                I[i, j, k_] = 1

            xy = np.max(I, axis = (2,))
            xz = np.max(I, axis = (1,))
            yz = np.max(I, axis = (0,))

            xyC = scipy.signal.correlate(xy, xy, mode="full")
            xzC = scipy.signal.correlate(xz, xz, mode="full")
            yzC = scipy.signal.correlate(yz, yz, mode="full")
            xyC[abs(xyC < 1e-8)] = 0
            xzC[abs(xzC < 1e-8)] = 0
            yzC[abs(yzC < 1e-8)] = 0

            f1 = np.count_nonzero(xyC) / xyC.shape[0] / xyC.shape[1]
            f2 = np.count_nonzero(xzC) / xzC.shape[0] / xzC.shape[1]
            f3 = np.count_nonzero(yzC) / yzC.shape[0] / yzC.shape[1]
            
            fill_factor.append(f1+f2+f3) # Save sum of three fill factors at this observation

            if plotting:
                # XY
                # On the first row we plot the Golomb Grids
                axs[k * self.n_meas].imshow(xy, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas].add_patch(plt.Rectangle((int(self.grid_size / 2)-0.5, int(self.grid_size / 2)-0.5), 1, 1, color='black', alpha=0.5))
                axs[k * self.n_meas].text( int(self.grid_size / 2), int(self.grid_size / 2), 'M', color='white', fontsize=12, ha='center', va='center')
                axs[k * self.n_meas].grid(False)
                axs[k * self.n_meas].set_xlim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas].set_ylim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas].axis("off")
                if k == 0:
                    axs[k * self.n_meas].set_title(f"1st snapshot\nt = 0\nXY plane\n{int(np.sum(xy))} remaining !", color="black")
                elif k == 1:
                    axs[k * self.n_meas].set_title(f"2nd snapshot\nt = 1 period\nXY plane\n{int(np.sum(xy))} remaining !", color="black")
                else:
                    axs[k * self.n_meas].set_title(f"3rd snapshot\nt = 2 periods\nXY plane\n{int(np.sum(xy))} remaining !", color="black")


                # On the second row we plot the autocorrelated Golomb Grids
                axs[k * self.n_meas + 3 * self.n_meas].imshow(xyC, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas + 3 * self.n_meas].grid(False)
                axs[k * self.n_meas + 3 * self.n_meas].axis("off")
                if k == 0:
                    axs[k * self.n_meas + 3 * self.n_meas].set_title(f"{f1:1.6f}", color="red")
                elif k == 1:
                    axs[k * self.n_meas + 3 * self.n_meas].set_title(f"{f1:1.6f}", color="red")
                else:
                    axs[k * self.n_meas + 3 * self.n_meas].set_title(f"{f1:1.6f}", color="red")

                # XZ
                # On the first row we plot the Golomb Grids
                axs[k * self.n_meas + 1].imshow(xz, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas + 1].add_patch(plt.Rectangle((int(self.grid_size / 2)-0.5, int(self.grid_size / 2)-0.5), 1, 1, color='black', alpha=0.5))
                axs[k * self.n_meas + 1].text( int(self.grid_size / 2), int(self.grid_size / 2), 'M', color='white', fontsize=12, ha='center', va='center')
                axs[k * self.n_meas + 1].grid(False)
                axs[k * self.n_meas + 1].set_xlim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas + 1].set_ylim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas + 1].axis("off")
                if k == 0:
                    axs[k * self.n_meas + 1].set_title(f"1st snapshot\nt = 0\nXZ plane\n{int(np.sum(xz))} remaining !", color="black")
                elif k == 1:
                    axs[k * self.n_meas + 1].set_title(f"2nd snapshot\nt = 1 period\nXZ plane\n{int(np.sum(xz))} remaining !", color="black")
                else:
                    axs[k * self.n_meas + 1].set_title(f"3rd snapshot\nt = 2 periods\nXZ plane\n{int(np.sum(xz))} remaining !", color="black")

                # On the secnd raw we plot the autocorrelated Golomb Grids
                axs[k * self.n_meas + 1 + 3 * self.n_meas].imshow(xzC, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas + 1 + 3 * self.n_meas].grid(False)
                axs[k * self.n_meas + 1 + 3 * self.n_meas].axis("off")
                if k == 0:
                    axs[k * self.n_meas + 1 + 3 * self.n_meas].set_title(f"{f2:1.6f}", color="red")
                elif k == 1:
                    axs[k * self.n_meas + 1 + 3 * self.n_meas].set_title(f"{f2:1.6f}", color="red")
                else:
                    axs[k * self.n_meas + 1 + 3 * self.n_meas].set_title(f"{f2:1.6f}", color="red")

                # XZ
                # On the first raw we plot the Golomb Grids
                axs[k * self.n_meas + 2].imshow(yz, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas + 2].add_patch(plt.Rectangle((int(self.grid_size / 2)-0.5, int(self.grid_size / 2)-0.5), 1, 1, color='black', alpha=0.5))
                axs[k * self.n_meas + 2].text( int(self.grid_size / 2), int(self.grid_size / 2), 'M', color='white', fontsize=12, ha='center', va='center')
                axs[k * self.n_meas + 2].grid(False)
                axs[k * self.n_meas + 2].set_xlim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas + 2].set_ylim(-0.5, self.grid_size - 0.5)
                axs[k * self.n_meas + 2].axis("off")

                if k == 0:
                    axs[k * self.n_meas + 2].set_title(f"1st snapshot\nt = 0\nYZ plane\n{int(np.sum(yz))} remaining !", color="black")
                elif k == 1:
                    axs[k * self.n_meas + 2].set_title(f"2nd snapshot\nt = 1 period\nYZ plane\n{int(np.sum(yz))} remaining !", color="black")
                else:
                    axs[k * self.n_meas + 2].set_title(f"3rd snapshot\nt = 2 periods\nYZ plane\n{int(np.sum(yz))} remaining !", color="black")

                # On the secnd raw we plot the autocorrelated Golomb Grids
                axs[k * self.n_meas + 2 + 3 * self.n_meas].imshow(yzC, cmap=cm.jet, interpolation="nearest", origin='lower')
                axs[k * self.n_meas + 2 + 3 * self.n_meas].grid(False)
                axs[k * self.n_meas + 2 + 3 * self.n_meas].axis("off")
                if k == 0:
                    axs[k * self.n_meas + 2 + 3 * self.n_meas].set_title(f"{f3:1.6f}", color="red")
                elif k == 1:
                    axs[k * self.n_meas + 2 + 3 * self.n_meas].set_title(f"{f3:1.6f}", color="red")
                else:
                    axs[k * self.n_meas + 2 + 3 * self.n_meas].set_title(f"{f3:1.6f}", color="red")

        return [-min(fill_factor)] # Return worst of all three observations

    
    def getRemaining(self, x):
        """
        Extracts the number of remaining satellites at each measurement point without plotting.

        Args:
            x (`list` of length N): Chromosome contains initial relative positions and velocities of each satellite.

        Returns:
            list: A list of integers representing the number of remaining satellites at each measurement point.
        """
        remaining_satellites = []

        # 1) Decode the chromosomes into (x, y, z, vx, vy, vz) of the satellites.
        N = self.n_sat
        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[:N], 
                                                            x[N:2*N], 
                                                            x[2*N:3*N],
                                                            x[3*N:4*N],
                                                            x[4*N:5*N],
                                                            x[5*N:])]
        )

        # 2) Propagate positions to the measurement points
        rel_pos = []
        for stm in self.stms:
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            rel_pos.append(fc / self.scaling_factor)
        rel_pos = np.array(rel_pos)

        # 3) Compute the number of remaining satellites at each observation point
        for k in range(self.n_meas):
            points_3D = rel_pos[k]
            
            # Apply inflation factor if not the first measurement
            if k != 0:
                points_3D = points_3D / self.inflation_factor

            # Remove points outside the [-1,1] range
            points_3D = points_3D[np.max(points_3D, axis=1) < 1]
            points_3D = points_3D[np.min(points_3D, axis=1) > -1]

            # Interpret the positions as grid points
            pos3D = (points_3D * self.grid_size / 2).astype(int)
            pos3D = pos3D + int(self.grid_size / 2)
            
            # Count the remaining satellites by summing unique positions
            remaining_satellites.append(len(pos3D))

        return remaining_satellites


    def getLost(self, x):
        """
        Extracts the indices of the lost satellites at each measurement point.

        Args:
            x (`list` of length N): Chromosome contains initial relative positions and velocities of each satellite.

        Returns:
            list of lists: A list containing three lists, each with the indices of the satellites that are not remaining 
            at the corresponding measurement point.
        """
        lost_satellites = []

        # 1) Decode the chromosomes into (x, y, z, vx, vy, vz) of the satellites.
        N = self.n_sat
        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[:N], 
                                                            x[N:2*N], 
                                                            x[2*N:3*N],
                                                            x[3*N:4*N],
                                                            x[4*N:5*N],
                                                            x[5*N:])]
        )

        # 2) Propagate positions to the measurement points
        rel_pos = []
        for stm in self.stms:
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            rel_pos.append(fc / self.scaling_factor)
        rel_pos = np.array(rel_pos)

        # 3) Find the lost satellites at each observation point
        for k in range(self.n_meas):
            points_3D = rel_pos[k]
            
            # Apply inflation factor if not the first measurement
            if k != 0:
                points_3D = points_3D / self.inflation_factor

            # Remove points outside the [-1,1] range
            remaining_points = points_3D[np.max(points_3D, axis=1) < 1]
            remaining_points = remaining_points[np.min(remaining_points, axis=1) > -1]

            # Find the indices of remaining satellites
            remaining_indices = np.where(np.isin(points_3D, remaining_points))[0]

            # Get the indices of lost satellites by finding the difference
            all_indices = np.arange(N)
            lost_indices = np.setdiff1d(all_indices, remaining_indices)

            # Append lost indices to the list
            lost_satellites.append(lost_indices.tolist())

        return lost_satellites



    def get_fill_factors(self, x):
        """
        Extracts the fill factors of the XY, XZ, and YZ planes for each observation point.

        Args:
            x (`list` of length N): Chromosome containing initial relative positions and velocities of each satellite.

        Returns:
            list: A list of 9 fill factors corresponding to the XY, XZ, and YZ planes for each observation point.
                [XY0, XZ0, YZ0, XY1, XZ1, YZ1, XY2, XZ2, YZ2]
        """
        fill_factors = []

        # 1) Decode the chromosomes into (x, y, z, vx, vy, vz) of the satellites.
        N = self.n_sat
        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[:N], 
                                                            x[N:2 * N], 
                                                            x[2 * N:3 * N],
                                                            x[3 * N:4 * N],
                                                            x[4 * N:5 * N],
                                                            x[5 * N:])]
        )

        # 2) Propagate positions to the measurement points
        rel_pos = []
        for stm in self.stms:
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            rel_pos.append(fc / self.scaling_factor)
        rel_pos = np.array(rel_pos)

        # 3) Compute the fill factor for each observation
        for k in range(self.n_meas):
            points_3D = rel_pos[k]

            # Apply inflation factor if not the first measurement
            if k != 0:
                points_3D = points_3D / self.inflation_factor

            # Remove points outside the [-1,1] range
            points_3D = points_3D[np.max(points_3D, axis=1) < 1]
            points_3D = points_3D[np.min(points_3D, axis=1) > -1]

            # Interpret now the 3D positions [-1,1] as points on a grid.
            pos3D = (points_3D * self.grid_size / 2)
            pos3D = pos3D + int(self.grid_size / 2)
            pos3D = pos3D.astype(int)

            # Create the 3D array
            I = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            for i, j, k_ in pos3D:
                I[i, j, k_] = 1

            # Compute projections on XY, XZ, and YZ planes
            xy = np.max(I, axis=(2,))
            xz = np.max(I, axis=(1,))
            yz = np.max(I, axis=(0,))

            # Compute autocorrelations
            xyC = scipy.signal.correlate(xy, xy, mode="full")
            xzC = scipy.signal.correlate(xz, xz, mode="full")
            yzC = scipy.signal.correlate(yz, yz, mode="full")

            # Apply thresholds
            xyC[abs(xyC) < 1e-8] = 0
            xzC[abs(xzC) < 1e-8] = 0
            yzC[abs(yzC) < 1e-8] = 0

            # Compute fill factors
            f1 = np.count_nonzero(xyC) / xyC.shape[0] / xyC.shape[1]
            f2 = np.count_nonzero(xzC) / xzC.shape[0] / xzC.shape[1]
            f3 = np.count_nonzero(yzC) / yzC.shape[0] / yzC.shape[1]

            # Append the fill factors for XY, XZ, YZ for this observation
            fill_factors.extend([f1, f2, f3])

        return fill_factors


    def get_positions_at_times(self, x):
        """
        Given the initial conditions x for all satellites, returns their positions at t=0, t=1, and t=2 in x, y, z coordinates.

        Args:
            x (`list` of length N*6): Initial conditions for all satellites.
        
        Returns:
            numpy array: Positions at t=0, t=1, and t=2 for all satellites.
                        Shape is (3, N, 3), where each entry is [x, y, z] at a time.
        """
        N = self.n_sat
        dx0 = np.array(
            [(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(x[:N], 
                                                            x[N:2 * N], 
                                                            x[2 * N:3 * N],
                                                            x[3 * N:4 * N],
                                                            x[4 * N:5 * N],
                                                            x[5 * N:])]
        )

        positions = []
        for idx, stm in enumerate(self.stms[:3]):  # Only propagate for k=0,1,2
            d_ic = dx0 * self.scaling_factor
            fc = propagate_formation(d_ic, stm)
            pos = fc / self.scaling_factor
            positions.append(pos)
        
        return np.array(positions)  # Shape is (3, N, 3)

    def get_grid_positions_at_times(self, x):
        """
        Given the initial conditions x for all satellites, returns their positions at t=0, t=1, and t=2 in terms of grid indices.

        Args:
            x (`list` of length N*6): Initial conditions for all satellites.

        Returns:
            List[List[List[int]]]: Grid positions at t=0, t=1, and t=2 for all satellites.
                                Each entry is a list of [i, j, k] for each satellite.
        """
        positions_at_times = self.get_positions_at_times(x)  # Shape (3, N, 3)

        grid_positions = []
        for time_index, positions in enumerate(positions_at_times):
            if time_index != 0:
                positions = positions / self.inflation_factor

            grid_positions_at_time = []
            for point in positions:
                if np.all(point >= -1) and np.all(point <= 1):
                    pos3D = (point * self.grid_size / 2) + int(self.grid_size / 2)
                    pos3D = pos3D.astype(int)
                    grid_positions_at_time.append(pos3D.tolist())
                else:
                    grid_positions_at_time.append([None, None, None])
            
            grid_positions.append(grid_positions_at_time)
        
        return grid_positions  # Shape (3, N, 3)

    def get_grid_positions_at_times_single_satellite(self, x):
        """
        Given the initial conditions x for a single satellite, returns its positions at t=0, t=1, and t=2 in terms of grid indices.

        Args:
            x (list or numpy array): Initial conditions [x0, y0, z0, vx0, vy0, vz0] for the satellite.

        Returns:
            List[List[int]]: Grid positions at t=0, t=1, and t=2 for the satellite.
                            Each element is a list [i, j, k].
                            If the satellite is out of bounds at a time, the element contains [None, None, None].
        """
        # Get the positions at times
        positions_at_times = self.get_positions_at_times_single_satellite(x)  # positions_at_times is a numpy array of shape (3, 3)

        grid_positions = []
        for k, point_3D in enumerate(positions_at_times):
            if k != 0:
                point_3D = point_3D / self.inflation_factor

            # Check if the satellite is within bounds [-1, 1]
            if np.all(point_3D >= -1) and np.all(point_3D <= 1):
                # Map position to grid index
                pos3D = (point_3D * self.grid_size / 2) + int(self.grid_size / 2)
                pos3D = pos3D.astype(int)
                grid_positions.append(pos3D.tolist())  # Convert to list
            else:
                grid_positions.append([None, None, None])  # Use [None, None, None] instead of None

        return grid_positions  # Return as list of lists

    def get_positions_at_times_single_satellite(self, x):
        """
        Given the initial conditions x for a single satellite, returns its positions at t=0, t=1, and t=2 in x, y, z coordinates.

        Args:
            x (list or numpy array): Initial conditions [x0, y0, z0, vx0, vy0, vz0] for the satellite.

        Returns:
            numpy array: Positions at t=0, t=1, and t=2 for the satellite.
                        Shape is (3, 3), where each row is [x, y, z] at a time.
        """
        # Ensure x is a numpy array
        x = np.asarray(x)
        # Reshape x into (1, 6) to make it compatible with existing functions
        dx0 = x.reshape((1, 6))

        # Propagate positions to each measurement time
        positions = []
        for idx, stm in enumerate(self.stms[:3]):  # Only propagate for k=0, k=1, k=2
            # Scale the initial conditions
            d_ic = dx0 * self.scaling_factor
            # Propagate the formation
            fc = propagate_formation(d_ic, stm)
            # Store positions (rescaled back)
            pos = fc[0] / self.scaling_factor  # fc has shape (1, 3)
            positions.append(pos)
        
        positions_array = np.array(positions)  # Converts list of arrays to a 2D numpy array
        return positions_array  # Shape is (3, 3)






############### MEDIUM problem configuration

# DRO
ic = [0.896508460944940632764, 0., 0., 0.000000000000013951082, 0.474817948848534454598, 0.]
period = 2.6905181697222775

# Number of satellites
N = 40

# Grid size
grid_size = 21

############### Constants
# Number of observations
M = 3
T = period*(M-1) # This makes it so that each observation is made after each period

mu = 0.01215058560962404  # M_L/(M_T + M_L)

scaling_factor = 1e-4

inflation_factor = 1.23
###############

# Instantiate UDP
udp = orbital_golomb_array(n_sat=N, ic = ic, T = T, grid_size=grid_size, scaling_factor = scaling_factor, n_meas=M, inflation_factor = inflation_factor, mu=mu, verbose=False)
