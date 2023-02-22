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


    print("All tests passed")

    return




if __name__ == '__main__':

    parameters = {
        'time_start': 0,                                  
        'time_end': 100,
        'time_step': 0.01,
        'tolerance': 0.0001,
        'S0': 100,
        'R0': 10,
        'growth_rate_S': 1,
        'growth_rate_R': 1,
        'carrying_capacity': 100,
        'maximum_tollerated_dose': 1,
        'death_rate_S': 0.1,
        'death_rate_R': 0.1,
        'division_rate': 0.1,
        'therapy_type': 'notherapy'
    }

    test_parameters(parameters)