import numpy as np
from scipy import signal

class Transmitter:
    """
    Signal generator for respiratory patterns.

    This class generates time-discrete breathing signals such as eupnea,
    sighing, tachypnea, kussmaul, and cheyne-stokes. Amplitude and frequency
    scaling can be specified per pattern.

    :param amplitude: normalized Amplitude of the signal.
    :param frequency: Frequency of the signal in breaths per minute.
    :param sampling: Sampling rate in Hz.
    """

    def __init__(self, amplitude=1.0, frequency=15.0, sampling=1000):
        self.amplitude = amplitude
        self.frequency = frequency
        self.sampling = sampling

        self.current_amplitude = self.amplitude
        self.current_frequency = self.frequency
    
    # base signal
    def sinusoidal_signal(self, t, amplitude, frequency):
        """
        Generates a sinusoidal breathing signal.

        :param t (float): Time in seconds.
        :param amplitude (float): Signal amplitude.
        :param frequency (float): Signal frequency in breaths per minute.
        :return: Signal value at time t.
        """
        return amplitude * np.sin(2 * np.pi * frequency / 60 * t)

    def sawtooth_signal(self, t, amplitude, frequency):
        """
        Generates a sawtooth breathing signal.

        :param t (float): Time in seconds.
        :param amplitude (float): Signal amplitude.
        :param frequency (float): Signal frequency in breaths per minute.
        :return: Signal value at time t.
        """
        return amplitude * signal.sawtooth(2 * np.pi * frequency / 60 * t)

    def cubic_signal(self, duration=20, fs=1000, inhale_time=1.7, pause1_time=0.3, exhale_time=1.7, pause2_time=0.3):
        """
        Generates a breathing signal composed of cubic Hermite segments
        for inhalation, exhalation and pauses.

        :param duration: Total signal duration in seconds
        :param fs: Sampling frequency
        :param inhale_time: Inhalation phase duration
        :param pause1_time: Pause after inhalation
        :param exhale_time: Exhalation phase duration
        :param pause2_time: Pause after exhalation
        :return: Full cubic breathing signal as np.array
        """

        def hermite_segment(p0, p1, m0, m1, n):
            t = np.linspace(0, 1, n)
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        signal_all = []
        total_time = 0

        while total_time < duration:
            # Inhalation: von 1 → -1
            signal_all += list(hermite_segment(1, -1, m0=0, m1=0, n=int(inhale_time * fs)))
            total_time += inhale_time

            # Pause 1: konstant -1
            signal_all += list(hermite_segment(-1, -1, 0, 0, n=int(pause1_time * fs)))
            total_time += pause1_time

            # Exhalation: von -1 → 1
            signal_all += list(hermite_segment(-1, 1, m0=0, m1=0, n=int(exhale_time * fs)))
            total_time += exhale_time

            # Pause 2: konstant +1
            signal_all += list(hermite_segment(1, 1, 0, 0, n=int(pause2_time * fs)))
            total_time += pause2_time

        return np.array(signal_all)

    def gaussian_signal(self, duration=20, fs=100, inhale_time=1.7, pause1_time=0.3, exhale_time=1.7, pause2_time=0.3):
        """
        Generates a breathing signal based on Gaussian-shaped inhale/exhale waves
        with optional pause insertion events.

        :param duration: Total duration of the signal (in seconds)
        :param fs: Sampling frequency
        :param inhale_time: Duration of inhalation phase
        :param pause1_time: Duration of pause after inhalation
        :param exhale_time: Duration of exhalation phase
        :param pause2_time: Duration of pause after exhalation
        :param p_event: Probability for pause insertion event during inhale/exhale
        :return: 1D NumPy array of the breathing signal
        """
        
        def gaussian(time, fs):
            t = np.linspace(0, time, int(time * fs), endpoint=False)
            gauss_full = np.exp(-0.5 * (np.linspace(-3, 3, len(t)*2) ** 2))
            gauss_half = gauss_full[len(gauss_full)//2:]
            gauss_half = (gauss_half - gauss_half.min()) / (gauss_half.max() - gauss_half.min())
            return gauss_half

        def pause(time, fs, value):
            return value * np.ones(int(time * fs))

        signal_all = []
        total_time = 0

        while total_time < duration:
            # Inhalation (normalized from +1 to -1)
            inhale = -1 + 2 * gaussian(inhale_time, fs)
            signal_all.extend(inhale)
            total_time += inhale_time

            # Pause 1
            signal_all.extend(pause(pause1_time, fs, -1))
            total_time += pause1_time

            # Exhalation (normalized from -1 to +1)
            exhale = 1 - 2 * gaussian(exhale_time, fs)
            signal_all.extend(exhale)
            total_time += exhale_time

            # Pause 2
            signal_all.extend(pause(pause2_time, fs, 1))
            total_time += pause2_time

        return np.array(signal_all)

    # breathing patterns
    def apnea(self, t, **kwargs):
        """
        Generates an apnea signal (no breathing).

        :param t (float): Time in seconds.
        :return: Signal value at time t (always 0).
        """
        zeros = np.zeros_like(t)
        return zeros

    def eupnea(self, t, signal_type='sinus', amplitude_factor=1.0, frequency_factor=1.0, **kwargs):
        """
        Normal breathing pattern (eupnea).

        :param t (float): Time in seconds.
        :param signal_type (str): Type of waveform ('sinus', 'sawtooth', 'cubic', 'gaussian' or 'advanced').
        :param amplitude_factor (float): Scaling factor for the amplitude.
        :param frequency_factor (float): Scaling factor for the frequency.
        :return: Signal value at time t.
        """
        amplitude = self.amplitude * amplitude_factor
        self.current_amplitude = amplitude
        frequency = self.frequency * frequency_factor
        self.current_frequency = frequency

        if signal_type == 'sinus':
            return self.sinusoidal_signal(t, amplitude, frequency)
        elif signal_type == 'sawtooth':
            return self.sawtooth_signal(t, amplitude, frequency)
        elif signal_type == 'cubic':
            return self.cubic_signal(duration=kwargs.get('duration', 20), fs=kwargs.get('fs', 1000),
                                     inhale_time=kwargs.get('inhale_time', 1.7),
                                     pause1_time=kwargs.get('pause1_time', 0.3),
                                     exhale_time=kwargs.get('exhale_time', 1.7),
                                     pause2_time=kwargs.get('pause2_time', 0.3))
        elif signal_type == 'gaussian':
            return self.gaussian_signal(duration=kwargs.get('duration', 20), fs=kwargs.get('fs', 100),
                                        inhale_time=kwargs.get('inhale_time', 1.7),
                                        pause1_time=kwargs.get('pause1_time', 0.3),
                                        exhale_time=kwargs.get('exhale_time', 1.7),
                                        pause2_time=kwargs.get('pause2_time', 0.3))
        elif signal_type == 'advanced':
            print('TODO: Advanced signal not implemented.')
            return 0.0

    def tachypnea(self, t, signal_type='sinus', amplitude_factor=1.0, frequency_factor=1.0, **kwargs):
        """
        Rapid breathing pattern with increased frequency.

        :param t (float): Time in seconds.
        :param signal_type (str): Type of waveform ('sinus', 'sawtooth', or 'advanced').
        :param frequency_factor (float): Scaling factor for the frequency.
        :return: Signal value at time t.
        """
        return self.eupnea(t, signal_type=signal_type, amplitude_factor=amplitude_factor, frequency_factor=2*frequency_factor, **kwargs)

    def kussmaul(self, t, signal_type='sinus', amplitude_factor=1.0, frequency_factor=1.0, **kwargs):
        """
        Deep and labored breathing pattern (kussmaul).

        :param t (float): Time in seconds.
        :param signal_type (str): Type of waveform ('sinus', 'sawtooth', or 'advanced').
        :param amplitude_factor (float): Scaling factor for the amplitude.
        :param frequency_factor (float): Scaling factor for the frequency.
        :return: Signal value at time t.
        """
        return self.eupnea(t, signal_type=signal_type,
                           amplitude_factor=5 * amplitude_factor,
                           frequency_factor=1.5 * frequency_factor,
                           **kwargs)

    def sighing(self, t, signal_type='sinus', amplitude_factor=1.0, frequency_factor=1.0, **kwargs):
        """
        Breathing pattern with occasional deeper breaths.

        :param t (float): Time in seconds.
        :param signal_type (str): Type of waveform ('sinus', 'sawtooth', or 'advanced').
        :param amplitude_factor (float): Scaling factor for the amplitude.
        :param frequency_factor (float): Scaling factor for the frequency.
        :return: Signal value at time t.
        """
        return self.eupnea(t, signal_type=signal_type,
                           amplitude_factor=1.5*amplitude_factor,
                           frequency_factor=0.75*frequency_factor,
                           **kwargs)

    def cheyne_stokes(self, t, amplitude_factor=1.0, frequency_factor=1.0, breath_duration=20.0, apnea_duration=2.0, **kwargs):
        """
        Generates Cheyne-Stokes breathing pattern. Works with scalar or array input.

        :param t: Time in seconds (float or np.ndarray)
        :param amplitude_factor: Scaling factor for the amplitude.
        :param frequency_factor: Scaling factor for the frequency.
        :param breath_duration: Duration of the breathing phase in seconds.
        :param apnea_duration: Duration of the apnea phase in seconds.
        :return: Signal value(s) at time t.
        """
        is_scalar = np.isscalar(t)
        t = np.asarray(t)

        amplitude = self.amplitude * 4.5 * amplitude_factor
        frequency = self.frequency * frequency_factor
        cycle_duration = breath_duration + apnea_duration

        t_in_cycle = np.mod(t, cycle_duration)
        signal = np.zeros_like(t)

        # Identify time points during the breathing phase
        breath_mask = t_in_cycle >= apnea_duration
        t_breath = t_in_cycle[breath_mask] - apnea_duration
        half_duration = breath_duration / 2.0

        # Create envelope
        envelope = np.where(
            t_breath < half_duration,
            amplitude * (t_breath / half_duration),
            amplitude * (1 - (t_breath - half_duration) / half_duration)
        )

        sinus = self.sinusoidal_signal(t_breath, amplitude=1.0, frequency=frequency)
        signal[breath_mask] = envelope * sinus

        return signal.item() if is_scalar else signal

    # noise and distortions
    def add_noise(self, signal, noise_type='gaussian', **kwargs):
        """
        Add different types of noise to a signal.

        :param signal (array-like): Input signal to which noise will be added.
        :param noise_type (str): Type of noise to apply. One of:
            - 'gaussian': Adds Gaussian noise with standard deviation `std`.
            - 'uniform': Adds uniform noise between `low` and `high`.
            - 'salt_pepper': Randomly sets values to `amplitude` or 0 with probability `prob`.
            - 'quantization': Quantizes signal into `levels` discrete steps.
        :param kwargs: Additional parameters for noise generation:
            - For 'gaussian': `std` (default 0.1)
            - For 'uniform': `low` (default -0.1), `high` (default 0.1)
            - For 'salt_pepper': `prob` (default 0.01), `amplitude` (default 1.0)
            - For 'quantization': `levels` (default 16)
        :return: Noisy signal as NumPy array.
        :raises ValueError: If an unsupported noise type is specified.
        """
        signal = np.asarray(signal)

        if noise_type == 'gaussian':
            std = kwargs.get('std', 0.1)
            noise = np.random.normal(loc=0.0, scale=std, size=signal.shape)
            return signal + noise

        elif noise_type == 'uniform':
            low = kwargs.get('low', -0.1)
            high = kwargs.get('high', 0.1)
            noise = np.random.uniform(low=low, high=high, size=signal.shape)
            return signal + noise

        elif noise_type == 'salt_pepper':
            prob = kwargs.get('prob', 0.01)
            amplitude = kwargs.get('amplitude', 1.0)
            noisy = np.copy(signal)
            rnd = np.random.rand(*signal.shape)

            # Salt
            salt_mask = rnd < (prob / 2)
            noisy[salt_mask] = amplitude

            # Pepper
            pepper_mask = (rnd >= (prob / 2)) & (rnd < prob)
            noisy[pepper_mask] = 0.0

            return noisy

        elif noise_type == 'quantization':
            levels = kwargs.get('levels', 16)
            min_val, max_val = np.min(signal), np.max(signal)
            step = (max_val - min_val) / (levels - 1)
            quantized = np.round((signal - min_val) / step) * step + min_val
            return quantized

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def amplitude_distortions(self, signal, distortion_type='clipping', **kwargs):
        """
        Simulate amplitude distortions caused by the signal source or sensor.

        :param signal (float): Clean signal value.
        :param distortion_type (str): Type of distortion ('clipping', 'compression').
        :param kwargs: Parameters for distortion (e.g., max_val, response_curve).
        :return: Distorted signal value.
        """
        if distortion_type == 'clipping':
            max_val = kwargs.get('max_val', self.amplitude)
            min_val = kwargs.get('min_val', -self.amplitude)
            return np.clip(signal, min_val, max_val)

        elif distortion_type == 'compression':
            a = kwargs.get('a', self.amplitude)
            return a * np.tanh(signal / a)

        else:
            raise ValueError(f"Unknown distortion type: {distortion_type}")

    def baseline_drift(self, t, drift_type='linear', **kwargs):
        """
        Simulates baseline drift over time.

        :param t (float): Current time in seconds.
        :param drift_type (str): Type of drift ('linear', 'exponential', 'sinusoidal').
        :param kwargs: Parameters like rate, amplitude, etc.
        :return: Drift value at time t.
        """
        if drift_type == 'linear':
            rate = kwargs.get('rate', 0.01)  
            return rate * t

        elif drift_type == 'exponential':
            A = kwargs.get('A', 1.0)
            k = kwargs.get('k', 0.01)
            return A * (1 - np.exp(-k * t))

        elif drift_type == 'sinusoidal':
            A = kwargs.get('A', 0.2)
            f = kwargs.get('f', 1/300)  
            return A * np.sin(2 * np.pi * f * t)

        else:
            raise ValueError(f"Unknown drift type: {drift_type}")

    def motion_artifacts(self, value, artifact_type='spike', **kwargs):
        """
        Simulates motion artifacts caused by sensor movement or vibration.

        :param value (float): Clean signal value.
        :param artifact_type (str): Type of artifact ('spike', 'random_amplitude').
        :param kwargs: Parameters for artifact (e.g., prob, spike_amplitude).
        :return: Distorted signal value.
        """
        if artifact_type == 'spike':
            prob = kwargs.get('prob', 0.01)
            spike_amplitude = kwargs.get('spike_amplitude', self.amplitude * 2)
            if np.random.rand() < prob:
                direction = np.random.choice([-1, 1])
                return value + direction * spike_amplitude
            return value

        elif artifact_type == 'random_amplitude':
            prob = kwargs.get('prob', 0.01)
            min_val = kwargs.get('min_val', -self.amplitude)
            max_val = kwargs.get('max_val', self.amplitude)
            if np.random.rand() < prob:
                return np.random.uniform(min_val, max_val)
            return value

        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
    
    # molecular output
    def molecular_output(self, breath_signal, scale=1e6):
        """
        Converts a breathing signal into a molecular emission signal.

        :param breath_signal (array-like): Breathing signal values.
        :param scale (float): Scaling factor for the emission signal.
        :return: Emission signal as a numpy array.
        """
        breath_signal = np.asarray(breath_signal)
        emission = np.maximum(breath_signal, 0.0) * scale

        return emission