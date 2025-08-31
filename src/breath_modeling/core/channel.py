import numpy as np

class Channel:
    """
    Channel model for simulating the propagation of molecules in a medium.

    :param sampling_rate (int): Sampling rate in Hz, default is 1000.
    """

    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate

    def cir(self, t, M, D, r, v):
        """
        Passive receiver, advection-diffusion based channel model.

        :param t (array): Time vector
        :param M (int): Number of molecules
        :param D (float): Diffusion coefficient
        :param r (float): distance between transmitter and receiver
        :param v (float): Advection velocity
        :return: cir
        """
        t = np.asarray(t)
        cir = np.zeros_like(t, dtype=float)
        valid = t > 0
        cir[valid] = M / np.sqrt(4 * np.pi * D * t[valid]) * np.exp(-((r - v * t[valid]) ** 2) / (4 * D * t[valid]))
        return cir
    
    def attenuation(self, signal, t, **kwargs):
        """
        Apply attenuation based on distance, material, exponential decay,
        angle of incidence and airflow speed.

        :param signal: Input signal to be attenuated
        :param t: Time vector
        :param distance: Distance factor (optional)
        :param material_factor: Material factor (optional)
        :param decay_rate: Decay rate (optional)
        :param angle_of_incidence: Angle between exhalation and sensor (0° = direct, 90° = tangential)
        :param airflow_speed: Speed of ventilation or environmental wind
        :return: Attenuated signal
        """
        d = kwargs.get('distance', 1)
        m = kwargs.get('material_factor', 1)
        r = kwargs.get('decay_rate', 0.0)

        angle = kwargs.get('angle_of_incidence', 0.0)
        airflow = kwargs.get('airflow_speed', 0.0)  

        size = kwargs.get("obstacle_size", 0.0) # normalized [0,1]
        density = kwargs.get("obstacle_density", 0.0)  # 1.0 = dense (e.g. metal), 0.0 = no obstacle   

        # Core attenuations
        d = d / 100
        distance_att = 1.0 / (1.0 + d)
        material_att = m
        decay = np.exp(-r * t)

        # Angle attenuation: cos(0°)=1 (max), cos(90°)=0 (none)
        angle_rad = np.deg2rad(angle)
        angle_att = np.clip(np.cos(angle_rad), 0, 1)

        # Airflow effect (stronger airflow = stronger attenuation)
        airflow_att = 1.0 / (1.0 + airflow)
        
        # Obstacle effect (size and density)
        size_att = np.exp(-size)
        density_att = np.exp(-density)

        total_att = distance_att * material_att * decay * angle_att * airflow_att * size_att * density_att
        return signal * total_att
    
    def amplitude_dependent_noise(self, signal, noise_type='gaussian', **kwargs):
            """
            Add noise to the signal based on its amplitude.
            
            :param signal: Input signal to which noise will be added
            :param noise_type: Type of noise to add ('gaussian' or 'uniform')
            :param distance: Distance factor for noise scaling (default 0.3)
            :param base_strength: Base strength of the noise (default 0.01)
            :param scaling: Scaling type for noise ('linear' or 'quadratic', default 'linear')
            :return: Signal with added noise
            """

            distance = kwargs.get("distance", 0.3)
            base_strength = kwargs.get("base_strength", 0.01)
            scaling = kwargs.get("scaling", "linear") # 'linear' or 'quadratic'

            signal = np.asarray(signal)
            max_amp = np.max(np.abs(signal)) + 1e-8  # avoid division by zero
            normalized_amplitude = np.abs(signal) / max_amp

            # Calculate noise strength based on distance and normalized amplitude
            if scaling == "linear":
                noise_strength = base_strength * distance * normalized_amplitude
            elif scaling == "quadratic":
                noise_strength = base_strength * distance * (normalized_amplitude ** 2)
            else:
                raise ValueError(f"Unknown scaling type: {scaling}")
            
            # Add noise based on the specified type
            if noise_type == 'gaussian':
                noisy_signal = signal + np.random.normal(0, noise_strength)
            elif noise_type == 'uniform':
                noisy_signal = signal + np.random.uniform(-noise_strength, +noise_strength)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")

            return noisy_signal
     
    def noise_floor(self, signal, **kwargs):
        """
        Adds a noise floor to the signal, simulating the minimum detectable signal level.

        :param signal: Input signal
        :param noise_floor: Noise floor level (default 0.01)
        :return: Signal with noise floor applied
        """
        noise_floor = kwargs.get('noise_floor', 0.01)
        return signal + np.random.normal(0, noise_floor, len(signal))

    def sampling_jitter(self, signal, **kwargs):
        """
        Randomly perturbs sample positions to simulate sampling jitter caused by unstable clocks.

        :param signal: Input signal
        :param jitter_std: Standard deviation of the jitter
        :return: Jittered signal
        """
        std = kwargs.get('jitter_std', 0.001)

        jitter = np.random.normal(0, std, len(signal))
        indices = np.arange(len(signal)) + jitter * self.sampling_rate
        indices = np.clip(indices.astype(int), 0, len(signal) - 1)

        return signal[indices]

    def time_warping(self, signal, **kwargs):
        """
        Simulates smooth cumulative time distortion due to breathing irregularity or system instability.

        :param signal: Input signal
        :param time_warp_strength: Base warping factor
        :return: Warped signal
        """
        strength = kwargs.get('time_warp_strength', 0.0)

        if strength == 0:
            return signal

        warp_noise = np.random.normal(0, strength, len(signal))
        warped_indices = np.cumsum(1 + warp_noise)
        warped_indices -= warped_indices[0]
        warped_indices *= (len(signal) - 1) / warped_indices[-1]
        return np.interp(np.arange(len(signal)), warped_indices, signal)

    def dropped_samples(self, signal, **kwargs):
        """
        Randomly drops samples from the signal to simulate data loss or sensor dropout.
        :param signal: Input signal
        :param drop_prob: Probability of dropping each sample (0.0 to 1.0)
        :param drop_block: If > 1, drops blocks of samples instead of individual ones
        :return: Signal with dropped samples (NaN where samples are dropped)
        """
        prob = kwargs.get('drop_prob', 0.0)
        block_size = int(kwargs.get('drop_block', 1)) # int drauß machen
        dropped_signal = signal.copy()
        
        i = 0
        while i < len(signal):
            if np.random.rand() < prob:
                dropped_signal[i:i+block_size] = np.nan
                i += block_size
            else:
                i += 1
        return dropped_signal
    
    def global_delay(self, signal, **kwargs):
        """
        Applies a fixed delay to simulate latency effects (e.g., sensor response, particle travel time).
        
        :param signal: Input signal to delay
        :param delay: Delay duration (in seconds or samples), optional
        :param delay_in_seconds: If True, 'delay' is interpreted as time; otherwise as samples
        :return: Delayed signal (with wraparound or optional NaN padding)
        """
        delay = kwargs.get('delay', 0.0)
        delay_in_seconds = kwargs.get('delay_in_seconds', True)

        if delay_in_seconds:
            delay_samples = int(delay * self.sampling_rate)
        else:
            delay_samples = int(delay)

        if delay_samples == 0:
            return signal

        delayed = np.full_like(signal, np.nan)
        if delay_samples < len(signal):
            delayed[delay_samples:] = signal[:len(signal) - delay_samples]
        return delayed

    def amplitude_compression(self, signal, **kwargs):
        """
        Applies amplitude compression to the signal using the hyperbolic tangent function.

        :param signal: input signal
        :param compression_factor: compression factor (optional)
        :return: compressed signal
        """
        a = kwargs.get('compression_factor', 1)
        return np.tanh(a * signal)
    
    def cross_wind(self, signal, t=None, **kwargs):
        """
        Simulates crosswind effects on the signal.

        :param signal: Input signal
        :param t: Time vector (optional, defaults to evenly spaced samples)
        :param strength: Strength of the crosswind effect (default 0.0)
        :param frequency: Frequency of the crosswind effect (default 0.0)
        :param phase: Phase shift of the crosswind effect (default 0.0)
        :return: Signal with crosswind effect applied
        """
        if t is None:
            t = np.linspace(0, len(signal) / self.sampling_rate, len(signal))

        strength = kwargs.get('strength', 0.0)
        frequency = kwargs.get('frequency', 0.0)
        phase = kwargs.get('phase', 0.0)

        foreign = strength * np.sin(2 * np.pi * frequency * t + phase)
        return signal + foreign
    
    def add_foreign_signal(self, signal, t, foreign_signal=None, **kwargs):
        """
        Adds interference from a secondary breathing source (e.g. other person).

        :param signal: Primary breathing signal
        :param t: Time vector
        :param foreign_signal: Optional signal from another person (same length)
        :param strength: Scaling factor for foreign signal
        :param delay: Time delay (in seconds) of foreign signal
        :param shift_freq: Optional frequency shift (Hz)
        :return: Composite signal
        """
        strength = kwargs.get('strength', 0.3)
        delay = kwargs.get('delay', 0.0)
        shift_freq = kwargs.get('shift_freq', 0.0)

        if foreign_signal is None:
            freq = kwargs.get('foreign_freq', 0.25)
            foreign_signal = np.sin(2 * np.pi * freq * t)

        # Apply delay
        delay_samples = int(delay * self.sampling_rate)
        delayed = np.roll(foreign_signal, delay_samples)

        # frequency modulation
        if shift_freq > 0:
            delayed *= np.sin(2 * np.pi * shift_freq * t)

        return signal + strength * delayed