import numpy as np
from scipy.stats import expon
from scipy.optimize import minimize

def kaplan_meier(data_matrix) :
    num_failures = len(data_matrix)
    num_rows = len(data_matrix)

    # Separate the data into observed times and event indicators
    observed_times = np.array(data_matrix)
    event_indicators = np.ones(num_failures)

    # Sort the observed times in ascending order
    sorted_times = np.sort(observed_times)

    # Initialize the Kaplan - Meier estimate
    km_estimate = np.ones(num_failures)

    # Compute the Kaplan - Meier estimate
    for i in range(num_failures) :
        if event_indicators[i] == 1 :
            km_estimate[i + 1:] *= (1 - 1 / (num_rows - i))

            return km_estimate

            def nelson_aalen(data_matrix) :
            num_failures = len(data_matrix)
            num_rows = len(data_matrix)

            # Separate the data into observed times and event indicators
            observed_times = np.array(data_matrix)
            event_indicators = np.ones(num_failures)

            # Sort the observed times in ascending order
            sorted_times = np.sort(observed_times)

            # Initialize the Nelson - Aalen estimate
            na_estimate = np.zeros(num_failures)

            # Compute the Nelson - Aalen estimate
            for i in range(num_failures) :
                if event_indicators[i] == 1 :
                    na_estimate[i + 1:] += 1 / (num_rows - i)

                    return na_estimate

                    # Example data matrix
                    data_matrix = [10, 22, 19, 0, 18, 7, 6, 13, 4, 14, 5, 10, 24,
                    4, 9, 23, 4, 3, 16, 11, 11, 7, 20, 7, 21, 1, 23,
                    10, 24, 7, 15, 2, 12, 8, 17, 14, 9, 10, 2, 11, 23,
                    20, 16, 8, 8, 10, 24, 23, 22, 10, 0, 1, 1, 0, 1,
                    1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
                    0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
                    1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0]

                    # Compute Kaplan - Meier and Nelson - Aalen estimates
                    km_estimate = kaplan_meier(data_matrix)
                    na_estimate = nelson_aalen(data_matrix)

                    # Print the estimates
                    print("Kaplan-Meier Estimate:")
                    print(km_estimate)

                    print("\nNelson-Aalen Estimate:")
                    print(na_estimate)
