import numpy as np
import matplotlib.pyplot as plt



def test_parameters(parameters):

    time_start = parameters['time_start']
    time_end = parameters['time_end']
    time_step = parameters['time_step']
    tolerance = parameters['tolerance']
    S0 = parameters['S0']
    R0 = parameters['R0']
    growth_rate_S = parameters['growth_rate_S']
    growth_rate_R = parameters['growth_rate_R']
    carrying_capacity = parameters['carrying_capacity']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']
    death_rate_S = parameters['death_rate_S']
    death_rate_R = parameters['death_rate_R']
    division_rate = parameters['division_rate']
    therapy_type = parameters['therapy_type']
    current_state = parameters['current_state']
    adaptive_therapy_ratio = parameters['adaptive_therapy_ratio']
    
    assert time_start >= 0
    assert time_end > time_start
    assert time_step > 0
    assert tolerance > 0
    assert S0 >= 0
    assert R0 >= 0
    assert growth_rate_S > 0
    assert growth_rate_R > 0
    assert carrying_capacity > 0
    assert maximum_tollerated_dose > 0
    assert death_rate_S > 0
    assert death_rate_S < 1
    assert death_rate_R > 0
    assert death_rate_R < 1
    assert division_rate > 0
    assert division_rate < 1
    assert therapy_type == 'continuous' or therapy_type == 'adaptive' or therapy_type == 'notherapy'
    assert adaptive_therapy_ratio > 0
    assert adaptive_therapy_ratio < 1
    assert current_state == 0 or current_state == 1


    print("All tests passed")

    return


def therapy_drug_concentration(N, parameters,k):

    therapy_type = parameters['therapy_type']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']

    if therapy_type == 'continuous':
        return maximum_tollerated_dose
    if therapy_type == 'notherapy':
        return 0
    elif therapy_type == 'adaptive':
        N0 = parameters['S0'] + parameters['R0']
        adaptive_therapy_ratio = parameters['adaptive_therapy_ratio']
        current_state = parameters['current_state']
        # on therapy
        if current_state > 0.001:
            # if shrunk sufficiently, turn off therapy slowly
            if N < N0 * adaptive_therapy_ratio:
                parameters['current_state'] = current_state-k
                return parameters['current_state']
            else:
            # else, keep on therapy
                parameters['current_state'] = current_state+k
                if parameters['current_state']> maximum_tollerated_dose:
                    parameters['current_state']= maximum_tollerated_dose
                    return parameters['current_state']
                else:
                    return parameters['current_state']
        else:
            # if grown sufficiently, turn on therapy
            if N > N0:
                parameters['current_state'] = current_state+k
                return parameters['current_state']
            else:
            # else, keep off therapy
                return 0
        

def one_step(S, R, time_step, parameters,k):

    growth_rate_S = parameters['growth_rate_S']
    growth_rate_R = parameters['growth_rate_R']
    carrying_capacity = parameters['carrying_capacity']
    maximum_tollerated_dose = parameters['maximum_tollerated_dose']
    death_rate_S = parameters['death_rate_S']
    death_rate_R = parameters['death_rate_R']
    division_rate = parameters['division_rate']

    N = S + R
    
    #We want carrying capacity to change with time!!!
    parameters['carrying_capacity'] = carrying_capacity
    
    current_carrying_capacity = N / parameters['carrying_capacity']
    
    D = therapy_drug_concentration(N, parameters, k)/(maximum_tollerated_dose)
    
    effective_growth_rate_S = growth_rate_S * (1 - current_carrying_capacity) * (1 - 2*division_rate*D)
    effective_growth_rate_R = growth_rate_R * (1 - current_carrying_capacity)

    dS = effective_growth_rate_S * S - death_rate_S * S
    dR = effective_growth_rate_R * R - death_rate_R * R

    S1 = S + dS * time_step
    R1 = R + dR * time_step

    return [S1, R1, D]


def two_step(S, R, time_step, parameters,k):
    
    SR1 = one_step(S, R, time_step/2, parameters,k)
    SR2 = one_step(SR1[0], SR1[1], time_step/2, parameters,k)

    return SR2



def ode_model(parameters, kappa, verbose=False):


    time_step = parameters['time_step']
    tolerance = parameters['tolerance']

    time_start = parameters['time_start']
    time_end = parameters['time_end']

    S0 = parameters['S0']
    R0 = parameters['R0']
    N0 = S0 + R0
    D0 = 1


    initial_size = int((time_end - time_start) / time_step)

    # Initialize arrays
    S = np.zeros(initial_size)
    R = np.zeros(initial_size)
    N = np.zeros(initial_size)
    T = np.zeros(initial_size)
    D = np.zeros(initial_size)

    current_time = time_start
    current_index = 0

    S[current_index] = S0
    R[current_index] = R0
    N[current_index] = N0
    T[current_index] = current_time
    D[current_index] = D0
    
    current_state = parameters['current_state']

    while current_time + time_step < time_end:

        
        SR_1 = one_step(S[current_index], R[current_index], time_step, parameters, kappa)
        SR_2 = two_step(S[current_index], R[current_index], time_step, parameters, kappa)
        
        error = abs(SR_1[0] - SR_2[0]) + abs(SR_1[1] - SR_2[1])

        if verbose == True:
            print("Error: ", error)
            print("Time step: ", time_step)
            print("Current time: ", current_time)
            print("Tolerance: ", tolerance)
            print("")

        if error < 10 * tolerance:
            S[current_index + 1] = SR_2[0]
            R[current_index + 1] = SR_2[1]
            N[current_index + 1] = S[current_index + 1] + R[current_index + 1]
            T[current_index + 1] = current_time + time_step
            D[current_index + 1] = SR_2[2]
            current_time += time_step
            current_index += 1
            # time_step = time_step/2
        elif error < tolerance:
            S[current_index + 1] = SR_2[0]
            R[current_index + 1] = SR_2[1]
            N[current_index + 1] = S[current_index + 1] + R[current_index + 1]
            T[current_index + 1] = current_time + time_step
            D[current_index + 1] = SR_2[2]
            current_time += time_step
            current_index += 1
        else:
            time_step = time_step/2
        
        if current_index == len(S) - 1:
            S = np.concatenate((S, np.zeros(len(S))))
            R = np.concatenate((R, np.zeros(len(R))))
            N = np.concatenate((N, np.zeros(len(N))))
            T = np.concatenate((T, np.zeros(len(T))))
            D = np.concatenate((D, np.zeros(len(D))))

    SR = two_step(S[current_index], R[current_index], time_end - current_time, parameters,kappa)
    S[current_index + 1] = SR[0]
    R[current_index + 1] = SR[1]
    N[current_index + 1] = S[current_index + 1] + R[current_index + 1]
    T[current_index + 1] = time_end
    current_index += 1

    return S[:current_index], R[:current_index], N[:current_index], T[:current_index], D[:current_index]
       

def main(initialRatio):
    # Define parameters

    parameters = {     
        'time_end': 1200,
        'time_start': 0,                                  
        'time_step': 0.1,
        'tolerance': 100,
        'R0': 0.002,
        'growth_rate_S': 0.023,
        'S0': 0.2,
        'growth_rate_R': 0.023,
        'carrying_capacity': 1.,
        'maximum_tollerated_dose': 1.,
        'death_rate_S': 0.01,
        'death_rate_R': 0.01,
        'division_rate': .75,
        'therapy_type': 'adaptive',
        'current_state': 1,
        'adaptive_therapy_ratio': 1/2,
    }
    total = parameters['S0'] + parameters['R0']
    parameters['R0'] = (initialRatio * total)/(1 + initialRatio)
    parameters['S0'] = (total) - parameters['R0']
    kappa = 1
    # test_parameters(parameters)

    Sc1, Rc1, Nc1, Tc1, Dc1 = ode_model(parameters, kappa, verbose=False)
    parameters['therapy_type'] = 'continuous'
    parameters['current_state'] = 1
    Sc2, Rc2, Nc2, Tc2, Dc2 = ode_model(parameters, kappa, verbose=False)

    plt.plot(Tc1, Sc1, label='S',linewidth=2)
    plt.plot(Tc1, Rc1, label='R',linewidth=2)
    plt.plot(Tc1, Nc1, label='N', linestyle='--',linewidth=2)
    plt.plot(Tc1, Dc1, label='D',linewidth=.7)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    filename = f'plot_varying_ratio={initialRatio}_adaptive.png'
    plt.savefig(filename)
    # clear plot
    plt.clf()

    plt.plot(Tc2, Sc2, label='S',linewidth=2)
    plt.plot(Tc2, Rc2, label='R',linewidth=2)
    plt.plot(Tc2, Nc2, label='N', linestyle='--',linewidth=2)
    plt.plot(Tc2, Dc2, label='D',linewidth=.7)
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.legend()
    filename = f'plot_varying_ratio={initialRatio}_continuous.png'
    plt.savefig(filename)
    # clear plot
    plt.clf()

    # compute time to reach 125% of initial tumor size
    initial_tumor_size = Nc1[0]
    for i in range(len(Nc1)):
        if Nc1[i] > 1.25*initial_tumor_size:
            ttpA = Tc1[i]
            break
    
    initial_tumor_size = Nc2[0]
    for i in range(len(Nc2)):
        if Nc2[i] > 1.25*initial_tumor_size:
            ttpC = Tc2[i]
            break
    
    print(f'Initial ratio: {initialRatio}')
    print(f'Time to reach 125% of initial tumor size (adaptive): {ttpA}')
    print(f'Time to reach 125% of initial tumor size (continuous): {ttpC}')
    gamma = (ttpA - ttpC)/ttpC
    print(f'Gamma: {gamma}')



if __name__ == '__main__':
    main(0.001)
    main(0.01)
    main(0.05)
    main(0.1)
    main(0.3)
    print('Done')