import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def normal_density_function(x, mean, standard_deviation):
    return np.exp(-(x - mean)**2 / (2 * standard_deviation**2)) / (standard_deviation * np.sqrt(2 * np.pi))


def initial_conditions(parameters):

    S0 = parameters['S0']
    R0 = parameters['R0']
    S0_distribution = parameters['S0_distribution']
    R0_distribution = parameters['R0_distribution']
    space_points = parameters['space_points']

    if S0_distribution == 'uniform':
        S = np.ones(space_points)
    elif S0_distribution == 'normal':
        space_start = parameters['space_start']
        space_end = parameters['space_end']
        X = np.linspace(space_start, space_end, space_points)
        standard_deviation_S = parameters['standard_deviation_S']
        mean = space_points//2
        # S is Gaussian distributed
        S = np.zeros(space_points)
        for i in range(space_points):
            S[i] = normal_density_function(X[i], mean, standard_deviation_S)
    elif S0_distribution == 'concentrated':
        S = np.zeros(space_points)
        S[space_points//2] = 1
        S[space_points//2 + 1] = 1
        S[space_points//2 - 1] = 1
    else:
        raise ValueError(
            'S0_distribution must be uniform or normal or concentrated')

    if R0_distribution == 'uniform':
        R = np.random.uniform(0, 1, space_points)
    elif R0_distribution == 'normal':
        space_start = parameters['space_start']
        space_end = parameters['space_end']
        X = np.linspace(space_start, space_end, space_points)
        standard_deviation_R = parameters['standard_deviation_R']
        mean = space_points//2
        # R is Gaussian distributed
        R = np.zeros(space_points)
        for i in range(space_points):
            R[i] = normal_density_function(X[i], mean, standard_deviation_R)
    elif R0_distribution == 'concentrated':
        R = np.zeros(space_points)
        R[space_points//2] = 1
        R[space_points//2 + 1] = 1
        R[space_points//2 - 1] = 1
    else:
        raise ValueError('R0_distribution must be uniform or normal')

    # rescale S and R to have the correct initial size
    S = S0 * S
    R = R0 * R
    N = S + R

    return S, R, N


def therapy_drug_concentration(S, R, parameters):

    N = np.trapz(S + R, dx=parameters['space_step'])

    therapy_type = parameters['therapy_type']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']

    if therapy_type == 'continuous':
        return maximum_tollerated_dose
    if therapy_type == 'notherapy':
        return 0
    elif therapy_type == 'adaptive':
        N0 = parameters['S0'] + parameters['R0']
        if N > 0.5 * N0:
            return maximum_tollerated_dose
        else:
            return 0
    else:
        raise ValueError(
            'therapy_type must be continuous, notherapy or adaptive')


def compute_diffusion(S, R, parameters):

    diffusion_type = parameters['diffusion_type']

    if diffusion_type == 'standard':
        space_step = parameters['space_step']
        diffusion_coefficient_S = parameters['diffusion_coefficient_S']
        diffusion_coefficient_R = parameters['diffusion_coefficient_R']
        diffusion_S = diffusion_coefficient_S * \
            (S[0] - 2 * S[1] + S[2]) / space_step**2
        diffusion_R = diffusion_coefficient_R * \
            (R[0] - 2 * R[1] + R[2]) / space_step**2
        
        if diffusion_S > 100000:
            print(diffusion_S)

        return diffusion_S, diffusion_R

    elif diffusion_type == 'none':
        return 0, 0
    elif diffusion_type == 'special':

        space_step = parameters['space_step']
        carrying_capacity_S = parameters['carrying_capacity_S']
        carrying_capacity_R = parameters['carrying_capacity_R']
        diffusion_coefficient_S = parameters['diffusion_coefficient_S']
        diffusion_coefficient_R = parameters['diffusion_coefficient_R']
        effective_diffusion_coefficient_S = diffusion_coefficient_S * \
            (1 - S[1] / carrying_capacity_S)
        effective_diffusion_coefficient_R = diffusion_coefficient_R * \
            (1 - R[1] / carrying_capacity_R)
        diffusion_S = effective_diffusion_coefficient_S * \
            (S[0] - 2 * S[1] + S[2]) / space_step**2
        diffusion_R = effective_diffusion_coefficient_R * \
            (R[0] - 2 * R[1] + R[2]) / space_step**2
        return diffusion_S, diffusion_R

    else:
        raise ValueError('diffusion_type must be standard, none or special')

def boundary_conditions(S_old, R_old, parameters):

    time_boundary_conditions = parameters['time_boundary_conditions']

    if time_boundary_conditions == 'Dirichlet':
        S_left = parameters['S0_left']
        R_left = parameters['R0_left']
        S_right = parameters['S0_right']
        R_right = parameters['R0_right']

    # Neumann boundary conditions read NSPDE on how to implement them
    elif time_boundary_conditions == 'Neumann':
        time_step = parameters['time_step']
        S_left = S_old[0]/2 + S_old[1]/2 + parameters['S0_left'] * time_step
        R_left = R_old[0]/2 + R_old[1]/2 + parameters['R0_left'] * time_step
        S_right = S_old[-1]/2 + S_old[-2]/2 + parameters['S0_right'] * time_step
        R_right = R_old[-1] + R_old[-2]/2 + parameters['R0_right'] * time_step

    else:
        raise ValueError(
            'time_boundary_conditions must be Dirichlet or Neumann')

    return S_left, R_left, S_right, R_right


def one_step(S_old, R_old, parameters):

    # initialize new arrays
    S = np.zeros(len(S_old))
    R = np.zeros(len(R_old))
    N = np.zeros(len(S_old))
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']

    # boundary conditions
    S[0], R[0], S[-1], R[-1] = boundary_conditions(S_old, R_old, parameters)
    N[0] = S[0] + R[0]
    N[-1] = S[-1] + R[-1]

    # constants
    time_step = parameters['time_step']
    growth_rate_S = parameters['growth_rate_S']
    growth_rate_R = parameters['growth_rate_R']
    carrying_capacity = parameters['carrying_capacity']
    division_rate = parameters['division_rate']
    death_rate_S = parameters['death_rate_S']
    death_rate_R = parameters['death_rate_R']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']
    D = therapy_drug_concentration(
        S_old, R_old, parameters)/maximum_tollerated_dose

    for i in range(1, len(S_old)-1):

        current_carrying_capacity = (S_old[i] + R_old[i]) / carrying_capacity
        effective_growth_rate_S = growth_rate_S * \
            (1 - current_carrying_capacity) * (1 - 2 * division_rate * D)
        effective_growth_rate_R = growth_rate_R * \
            (1 - current_carrying_capacity)

        dS = effective_growth_rate_S - death_rate_S
        dR = effective_growth_rate_R - death_rate_R

        diffusion_S, diffusion_R = compute_diffusion(
            S_old[i-1:i+2], R_old[i-1:i+2], parameters)

        S[i] = S_old[i] + time_step * (dS * S_old[i] + diffusion_S)
        R[i] = R_old[i] + time_step * (dR * R_old[i] + diffusion_R)

    tolerance = parameters['tolerance']
    mask_S = S < tolerance
    mask_R = R < tolerance
    S[mask_S] = 0
    R[mask_R] = 0

    return S, R, N, D


def pde_2D_model(parameters):

    time_start = parameters['time_start']
    time_end = parameters['time_end']
    time_points = parameters['time_points']
    parameters['time_step'] = (time_end - time_start) / time_points

    space_start = parameters['space_start']
    space_end = parameters['space_end']
    space_points = parameters['space_points']
    parameters['space_step'] = (space_end - space_start) / space_points

    X = np.linspace(space_start, space_end, space_points)
    T = np.linspace(time_start, time_end, time_points)

    S = np.ones((space_points, time_points))
    R = np.zeros((space_points, time_points))
    N = np.zeros((space_points, time_points))
    D = np.zeros(time_points)

    index = 0

    S[:, index], R[:, index], N[:, index] = initial_conditions(parameters)

    for t in T[1:]:
        S[:, index + 1], R[:, index + 1], N[:, index + 1], D[index +
                                                             1] = one_step(S[:, index], R[:, index], parameters)
        index += 1

    return S, R, N, D, X, T




if __name__ == '__main__':

    parameters = {
        'time_start': 0,
        'time_end': 100,
        'time_points': 10_000,
        'space_start': 0,
        'space_end': 0.5,
        'space_points': 100,
        'tolerance': 0.0000001,
        'S0': 1.4,
        'R0': 0.02,
        'S0_distribution': 'normal',
        'R0_distribution': 'normal',
        'growth_rate_S': 0.03,
        'growth_rate_R': 0.03,
        'carrying_capacity': 2,
        'diffusion_coefficient_S': 0.0001,
        'diffusion_coefficient_R': 0.0001,
        'standard_deviation_S': 0.01,
        'standard_deviation_R': 0.01,
        'maximum_tollerated_dose': 1,
        'death_rate_S': 0.03,
        'death_rate_R': 0.03,
        'division_rate': 0.4,
        'therapy_type': 'continuous',
        'time_boundary_conditions': 'Dirichlet',
        'S0_left': 0,
        'R0_left': 0,
        'S0_right': 0,
        'R0_right': 0,
        'diffusion_type': 'standard'
    }

    S, R, N, D, X, T = pde_2D_model(parameters)

    # plot the results
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('S')
    plotLine, = ax.plot(X, np.zeros(len(X))*np.NaN, 'r-')
    plotTitle = ax.set_title("t=0")
    ax.set_ylim(0, 2)
    ax.set_xlim(parameters['space_start'], parameters['space_end'])

    def animate(t):
        pp = S[:, t]
        plotLine.set_ydata(pp)
        plotTitle.set_text(f"t = {t:.1f}")
        # ax.relim() # use if autoscale desired
        # ax.autoscale()
        return [plotLine, plotTitle]

    ani = animation.FuncAnimation(
        fig, func=animate, frames=np.arange(len(T)), blit=True)

    WriterClass = animation.writers['ffmpeg']
    writer = animation.FFMpegFileWriter(fps=10, metadata=dict(artist='bww'), bitrate=1800)
    ani.save('pde_2D_model.mp4', writer = writer)