###################################################################################################
#   Filename:       VAB_FIR_Analysis_mi549_yrmz20.py
#   Description:    A waveform tool analysis performs FFT analysis on one line-to-line voltage signal (VAB) from one leg to another
#                   of a three phase inverter and carries out a corresponding designed hamming-windowed filter.
#   Date:           23/04/2026
#   Author:         Muhammad Izhar (mi549) & Yousef Zaidan (yrmz20)
###################################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt


class Waveform:
    def __init__(self, time_taken: list[float], voltage_signal: list[float]) -> None:
        """
        The waveform class represents a signal that is evenly sampled.

        Arguments:
        time_taken (list[float]): the time at which each sample is taken
        voltage_signal (list[float]): voltage level of each sample
        """
        if len(time_taken) != len(voltage_signal):
            raise ValueError(
                "The length of the voltage and time_taken must be the same")
        if len(time_taken) < 2:
            raise ValueError("Cannot have less than 2 samples")
        self.time_taken: list[float] = time_taken
        self.voltage_signal: list[float] = voltage_signal
        self.sampling_number: int = len(time_taken)
        # Sampling rate is calculated using the formula total span/(N-1) with the assumption of equal sampling rate
        self.fs: float = (len(time_taken) - 1) / \
            (time_taken[-1] - time_taken[0])

    def fft_analysis(self) -> tuple[list[float], list[float]]:
        """
        fft_analysis() uses the FFT technique to acquire a one-sided spectrum

        Returns:
            tuple[list[float], list[float]]: A tuple where the first element is the
            frequency measured in Hz, whereas the second element is the amplitude
            measured in Volts.
        """
        N = self.sampling_number
        result = np.fft.fft(np.array(self.voltage_signal))
        # Divide the results by N to normalise the data size
        peak_size = np.abs(result)/N
        N_half = N // 2
        # To compensate for folded energy, we multiply the amplitudes by 2
        one_side = 2 * peak_size[:N_half]
        bins = [x * self.fs / N for x in range(N_half)]
        # The bins have a spacing of fs/N < Nyquist limit
        return bins, one_side.tolist()

    def main_frequency(self) -> float:
        """
        main_frequency() locates the spectrum's dominant frequency

        Returns:
            float: Dominant peak measured in Hz.
        """
        frequency, peak_size = self.fft_analysis()
        peak_number = peak_size[1:].index(max(peak_size[1:])) + 1
        return frequency[peak_number]


class FIR_Filter:
    def __init__(self, number_of_order: int, cutoff_frequency: float, sampling_rate: float) -> None:
        """
        This class defines a low pass Windowed FIR filter with a linear phase response.
        A truncated version of the sinc impulse response and hamming window is used to
        reduce the stop-band ripple effect. The odd order filter ensures there is a symmetric tap structure,
        providing a group delay of (order -1)/2 samples
        Arguments:
            cutoff_frequency: The cutoff frequency measured in Hz.
            sampling_rate: The sampling frequency of the clean signal measured in Hz.
            number_of_order: Odd positive integer setting the number of taps used in the filter
        """
        if number_of_order <= 0 or number_of_order % 2 == 0:
            raise ValueError("Filter order can only be a positive odd number")
        if not 0 < cutoff_frequency < sampling_rate / 2:
            raise ValueError(
                "Must satisfy 0 < cutoff_frequency < Nyquist frequency condition")

        self.cutoff_frequency: float = cutoff_frequency
        self.sampling_rate: float = sampling_rate
        self.number_of_order: int = number_of_order
        self.taps: list[float] = self.filter_settings()

    def filter_settings(self) -> list[float]:
        """
        filter_settings() sets the coefficients in the FIR filter with the
        windowed configurations

        Returns:
            list[float]: The normalised version of the coefficients
        """
        a = self.number_of_order
        # The cut off frequency is normalised, following the sinc equation
        fc_normalised = self.cutoff_frequency/self.sampling_rate
        # To gain linear phase, we centre the impulse response
        midpoint = (a - 1)/2

        taps: list[float] = []
        for x in range(a):
            off_by = x - midpoint
            if off_by == 0:
                # sinc(0) limit = 2 x the normalised cut off frequency
                ideal_sinc = 2 * fc_normalised
            else:
                ideal_sinc = math.sin(2 * fc_normalised *
                                      off_by * math.pi) / (off_by * math.pi)
                # the ideal low-pass response formula

            hamming_window = 0.54 - 0.46 * math.cos(2 * x * math.pi / (a-1))
            # Hamming window provides 53dB attenuation
            taps.append(ideal_sinc * hamming_window)
        # the coefficients are normalised to DC gain of 1
        total_taps = sum(taps)
        return [T / total_taps for T in taps]

    def operate(self, signal: "Waveform") -> 'Waveform':
        """
        Arguments:
            signal (Waveform): The unfiltered output signal from the inverter

            Carries out convolution by using coefficients which clean the signal

        Returns:
        Waveform: The filtered signal with the updated voltage values
        """
        cleaned_version = np.convolve(
            signal.voltage_signal, self.taps, mode="same")
        return Waveform(time_taken=signal.time_taken, voltage_signal=cleaned_version.tolist())


def gather_LTSPICE_document(path: str) -> Waveform:
    """
    gather_LTSPICE_document() function reads the LTSPICE export file
    with two columns and proceeds with a signal output

    Arguments:
    path (str): Path to the exported LTSPICE file.

    Returns:
    Waveform: The unfiltered waveform object

    """

    Time: list[float] = []
    Voltage: list[float] = []

    with open(path, "r") as data_file:
        next(data_file)
        for every_line in data_file:
            section = every_line.split()
            if len(section) >= 2:
                Time.append(float(section[0]))
                Voltage.append(float(section[1]))

    return Waveform(time_taken=Time, voltage_signal=Voltage)


def graph_results(before: Waveform, after: Waveform) -> None:
    """
    graph_results() produces a time-domain and frequency-domain plots showing the before and after filtering signals graphs

    Arguments:

    before(Waveform): VAB before filtering
    after(Waveform): VAB after filtering


    """
    plt.figure()
    plt.plot(before.time_taken, before.voltage_signal,
             label="Before filtering VAB")
    plt.plot(after.time_taken, after.voltage_signal, label="Filtered VAB")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (V)")
    plt.title("Time Domain: VAB signal before/after filtering MI549 YRMZ20")
    plt.grid(True)
    plt.legend()

    before_frequency, a_before = before.fft_analysis()
    after_frequency, a_after = after.fft_analysis()

    plt.figure()
    plt.plot(before_frequency, a_before, label="Spectrum before filtering")
    plt.plot(after_frequency, a_after,
             label="Spectrum after filtering", linewidth=2)
    plt.xlabel("Wave Frequency (Hz)")
    plt.ylabel("Voltage Amplitude (V)")
    plt.title("Frequency Domain: VAB signal before/after filtering MI549 YRMZ20")
    plt.xlim(0, 10000)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    linevoltage_VAB = gather_LTSPICE_document("VAB_text")
    # The cut off frequency is 150Hz, which is higher than the 50Hz fundamental frequency but lower than the PWM switching frequencies,
    # separating the inverter's fundamental output
    filter_response = FIR_Filter(
        number_of_order=4001, cutoff_frequency=150, sampling_rate=linevoltage_VAB.fs)

    print(f"\nMuhammad Izhar (mi549) Yousef Zaidan (yrmz20)")
    print(f"\nWaveform signal has {linevoltage_VAB.sampling_number} samples")
    print(f"The sampling rate is {linevoltage_VAB.fs:.2f} Hz ")
    print(
        f"The dominant frequency is {linevoltage_VAB.main_frequency():.2f} Hz")

    filtered_waveform = filter_response.operate(linevoltage_VAB)

    print(
        f"\nFilter: Hamming window, fc = {filter_response.cutoff_frequency} Hz, N = {filter_response.number_of_order}")
    print(f"Main frequency peak: {filtered_waveform.main_frequency():.2f} Hz")
    graph_results(linevoltage_VAB, filtered_waveform)
