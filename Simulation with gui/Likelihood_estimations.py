
def likelihood_of_measurements_unrelated_to_emitter(associated_timings, emitter):
    """returns the probability of not detecting the specified associated timing from a known emitter.
        Args:
            associated_timings (list): list of the signal time measured by each sensor.
            the length of the list is the number of sensors.
            if the i'th sensor has no measure of the signal emitted by this emitter, the list will contain
            None in the i'th place.

            emitter: an emitter object specifying the signal and location of the emitter.

            time_error: the average of the gaussian time error.

            time_error_std: the standard deviation of the gaussian time error

        Returns:
            double - returns the probability of this state.
    """

def negative_log_likelihood_ratio(associated_timings, emitter):
    """returns the negative log-likelihood ratio of specified associated timing from a known emitter.
            Args:
                associated_timings (list): list of the signal time measured by each sensor.
                the length of the list is the number of sensors.
                if the i'th sensor has no measure of the signal emitted by this emitter, the list will contain
                None in the i'th place.

                emitter: an emitter object specifying the signal and location of the emitter.

                time_error: the average of the gaussian time error.

                time_error_std: the standard deviation of the gaussian time error

            Returns:
                double - returns the probability of this state.
        """

