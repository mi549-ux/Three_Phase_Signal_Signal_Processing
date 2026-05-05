##############################################################################################################################
#   Filename:       VAB_FIR_Analysis_mi549_yrmz20.py
#   Description:    This waveform analysis tool reads the line-to-line inverter voltage signal VAB from an LTspice export,
#                   performs FFT analysis to identify the main frequency components, and applies a designed
#                   Hamming-windowed FIR low-pass filter to single out the fundamental output waveform.
#                   Generating 5 plots that linked to the frequency-domain and time-domain plots without and with filtering.
#   Date:           23/04/2026
#   Author:         Muhammad Izhar (mi549) & Yousef Zaidan (yrmz20)
##############################################################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AnalysisReport:
    """
    The AnalysisReport class stores the calculated results from the VAB
    waveform analysis so that they can be printed or saved to a text file.

    Arguments:
        sampling_rate (float): Sampling rate of the waveform in Hz.
        sampling_number (int): Number of samples in the waveform.
        main_frequency_before (float): Dominant frequency before filtering in Hz.
        main_frequency_after (float): Dominant frequency after filtering in Hz.
        rms_before (float): RMS voltage before filtering.
        rms_after (float): RMS voltage after filtering.
        peak_to_peak_before (float): Peak-to-peak voltage before filtering.
        peak_to_peak_after (float): Peak-to-peak voltage after filtering.
        harmonic_amplitudes_before (dict[float, float]): Harmonic amplitudes before filtering.
        harmonic_amplitudes_after (dict[float, float]): Harmonic amplitudes after filtering.
    """
    sampling_rate: float
    sampling_number: int
    main_frequency_before: float
    main_frequency_after: float
    rms_before: float
    rms_after: float
    peak_to_peak_before: float
    peak_to_peak_after: float
    harmonic_amplitudes_before: dict[float, float]
    harmonic_amplitudes_after: dict[float, float]

    def save_to_text_file(self, path: Path) -> None:
        """
        save_to_text_file() writes the main analysis results to a text file.

        Arguments:
            path (Path): Location where the analysis report should be saved.
        """
        with open(path, "w") as report_file:
            report_file.write("VAB FIR Analysis Report\n")
            report_file.write("=======================\n\n")

            report_file.write(f"Sampling rate: {self.sampling_rate:.2f} Hz\n")
            report_file.write(f"Number of samples: {self.sampling_number}\n\n")

            report_file.write(
                f"Main frequency before filtering: {self.main_frequency_before:.2f} Hz\n")
            report_file.write(
                f"Main frequency after filtering: {self.main_frequency_after:.2f} Hz\n\n")

            report_file.write(
                f"RMS voltage before filtering: {self.rms_before:.2f} V\n")
            report_file.write(
                f"RMS voltage after filtering: {self.rms_after:.2f} V\n\n")

            report_file.write(
                f"Peak-to-peak voltage before filtering: {self.peak_to_peak_before:.2f} V\n")
            report_file.write(
                f"Peak-to-peak voltage after filtering: {self.peak_to_peak_after:.2f} V\n\n")

            report_file.write("Harmonic amplitudes before filtering:\n")
            for frequency, amplitude in self.harmonic_amplitudes_before.items():
                report_file.write(f"{frequency:.0f} Hz: {amplitude:.2f} V\n")

            report_file.write("\nHarmonic amplitudes after filtering:\n")
            for frequency, amplitude in self.harmonic_amplitudes_after.items():
                report_file.write(f"{frequency:.0f} Hz: {amplitude:.2f} V\n")


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

    def rms_voltage(self) -> float:
        """
        rms_voltage() calculates the root mean square voltage of the waveform.

        Returns:
            float: RMS voltage of the waveform.
        """
        return float(np.sqrt(np.mean(np.square(self.voltage_signal))))

    def peak_to_peak_voltage(self) -> float:
        """
        peak_to_peak_voltage() calculates the difference between the maximum
        and minimum voltage values in the waveform.

        Returns:
            float: Peak-to-peak voltage of the waveform.
        """
        return max(self.voltage_signal) - min(self.voltage_signal)

    def amplitude_at_frequency(self, target_frequency: float) -> float:
        """
        amplitude_at_frequency() finds the FFT amplitude closest to a selected
        target frequency.

        Arguments:
            target_frequency (float): Frequency to be checked in Hz.

        Returns:
            float: Amplitude at the closest available FFT frequency bin.
        """
        frequency, peak_size = self.fft_analysis()
        closest_index = min(
            range(len(frequency)),
            key=lambda index: abs(frequency[index] - target_frequency)
        )
        return peak_size[closest_index]


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


def gather_LTSPICE_document(path: str | Path) -> Waveform:
    """
    gather_LTSPICE_document() function reads the LTSPICE export file
    with two columns and proceeds with a signal output

    Arguments:
    path (str | Path): Path to the exported LTSPICE file.

    Returns:
    Waveform: The unfiltered waveform object
    """

    Time_without_filter: list[float] = []
    Voltage_without_filter: list[float] = []

    with open(path, "r") as data_file:
        next(data_file)
        for every_line in data_file:
            section = every_line.split()
            if len(section) >= 2:
                Time_without_filter.append(float(section[0]))
                Voltage_without_filter.append(float(section[1]))

    sample_period = 10e-6  # equivalent to 10 microseconds
    window_start = 0.140  # ignore transient response
    window_finish = 0.200  # end of the simulation period
    time_resampled = np.arange(window_start, window_finish, sample_period)
    voltage_resampled = np.interp(
        time_resampled, Time_without_filter, Voltage_without_filter)

    return Waveform(time_taken=time_resampled.tolist(), voltage_signal=voltage_resampled.tolist())


def create_analysis_report(before: Waveform, after: Waveform, harmonic_targets: list[float]) -> AnalysisReport:
    """
    create_analysis_report() produces a stored summary of the main waveform
    results before and after filtering.

    Arguments:
        before (Waveform): VAB waveform before filtering.
        after (Waveform): VAB waveform after filtering.
        harmonic_targets (list[float]): Frequencies to check in the FFT spectrum.

    Returns:
        AnalysisReport: Summary of the calculated waveform results.

    """
    before_harmonics: dict[float, float] = {}
    after_harmonics: dict[float, float] = {}

    for target in harmonic_targets:
        before_harmonics[target] = before.amplitude_at_frequency(target)
        after_harmonics[target] = after.amplitude_at_frequency(target)

    return AnalysisReport(
        sampling_rate=before.fs,
        sampling_number=before.sampling_number,
        main_frequency_before=before.main_frequency(),
        main_frequency_after=after.main_frequency(),
        rms_before=before.rms_voltage(),
        rms_after=after.rms_voltage(),
        peak_to_peak_before=before.peak_to_peak_voltage(),
        peak_to_peak_after=after.peak_to_peak_voltage(),
        harmonic_amplitudes_before=before_harmonics,
        harmonic_amplitudes_after=after_harmonics
    )


def graph_results(before: Waveform, after: Waveform) -> None:
    """
    graph_results() produces time-domain and frequency-domain plots showing
    the VAB signal before and after filtering.

            Graph 1 - Unprocessed Signal in Time Domain  
            Graph 2 - Linear Amplitude Spectrum
            Graph 3 - Log-log Amplitude Spectrum
            Graph 4 - Before & After Filtering Time Domain
            Graph 5 - Before & After Filtering Frequency Domain

    Arguments:
        before (Waveform): VAB before filtering.
        after (Waveform): VAB after filtering.
    """
    time_in_ms = [t * 1e3 for t in before.time_taken]
    before_frequency, amplitude_before = before.fft_analysis()
    after_frequency, amplitude_after = after.fft_analysis()

    # Graph 1 shows the raw signal in time-domain

    graph1, axes1 = plt.subplots()
    axes1.plot(time_in_ms, before.voltage_signal,
               color="blue", linewidth=0.5)
    axes1.set_xlabel("Time (ms)")
    axes1.set_ylabel("Voltage (V)")
    axes1.set_title(
        "V(A,B) - RAW PWM Three Phase Inverter Line-to-Line Voltage \nmi549 yrmz20 ")
    axes1.set_ylim([-400, 400])
    axes1.grid(True)
    graph1.tight_layout()

    # Graph 2 presents the Amplitude Spectrum using Linear Scale

    graph2, axes2 = plt.subplots()
    axes2.plot(before_frequency, amplitude_before,
               color="blue", linewidth=0.5)
    axes2.set_xlabel("Frequency (Hz)")
    axes2.set_ylabel("Amplitude (V)")
    axes2.set_title(
        "The Amplitude Spectrum for V(A,B) on Linear Scale\nmi549 yrmz20")
    axes2.set_xlim([0, 10000])
    axes2.grid(True)
    graph2.tight_layout()

    # Graph 3 presents the Amplitude Spectrum using Log-Log Scale

    log_frequency = before_frequency[1:]  # removes the DC bin eliminate log(0)
    log_amplitude = amplitude_before[1:]
    graph3, axes3 = plt.subplots()
    axes3.loglog(log_frequency, log_amplitude, color="blue", linewidth=0.5)
    axes3.set_xlabel("Frequency (Hz)")
    axes3.set_ylabel("Amplitude (V)")
    axes3.set_title("The Log-Log Spectrum for V(A,B) on Linear\nmi549 yrmz20")
    axes3.grid(True)
    graph3.tight_layout()

    # Graph 4 represents pre and post filteirng time domain plots

    graph4, (axes4_above, axes4_below) = plt.subplots(2, 1, figsize=(10, 7))

    axes4_above.plot(time_in_ms, before.voltage_signal,
                     color="blue", linewidth=0.5)
    axes4_above.set_xlabel("Time (ms)")
    axes4_above.set_ylabel("Voltage (V)")
    axes4_above.set_title(
        "BEFORE FILTERING V(A,B) PWM SIGNAL\nmi549 yrmz20 ")
    axes4_above.set_ylim([-400, 400])
    axes4_above.grid(True)

    axes4_below.plot(time_in_ms, after.voltage_signal,
                     color="red", linewidth=1)
    axes4_below.set_xlabel("Time (ms)")
    axes4_below.set_ylabel("Voltage (V)")
    axes4_below.set_title(
        " FILTERED V(A,B) SIGNAL WITH FUNDAMENTAL FREQUENCY RECOVERED\nmi549 yrmz20")
    axes4_below.set_ylim([-400, 400])
    axes4_below.grid(True)
    graph4.tight_layout()

    #  graph 5 represents pre and post filteirng time domain plots

    graph5, (axes5_above, axes5_below) = plt.subplots(2, 1, figsize=(10, 7))

    axes5_above.plot(before_frequency, amplitude_before,
                     color="blue", linewidth=0.5)
    axes5_above.set_xlabel("Frequency (Hz)")
    axes5_above.set_ylabel("Amplitude (V)")
    axes5_above.set_title(
        "PRIOR TO FILTERING THE AMPLITUDE SPECTRUM, INCLUDING 50 Hz and HARMONICS\nmi549 yrmz20 ")
    axes5_above.set_xlim([0, 10000])
    axes5_above.grid(True)

    axes5_below.plot(after_frequency, amplitude_after,
                     color="red", linewidth=1)
    axes5_below.set_xlabel("Frequency (Hz)")
    axes5_below.set_ylabel("Amplitude (V)")
    axes5_below.set_title(
        "FREQUENCY DOMAIN SPECTRUM OF THE FILTERED SIGNAL WITH ONLY 50 Hz\nmi549 yrmz20")
    axes5_below.set_xlim([0, 500])
    axes5_below.grid(True)
    graph5.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = Path(__file__).with_name("VAB_text")
    linevoltage_VAB = gather_LTSPICE_document(file_path)

    # The 150 Hz cut off frequency keeps the 50 Hz fundamental while removing the PWM switching region.
    # The 4001 tap order gives an odd, symmetric FIR filter structure for the low-pass filter.
    cutoff_frequency = 150
    number_of_order = 4001
    harmonic_targets = [50, 3000, 6000]

    filter_response = FIR_Filter(
        number_of_order=number_of_order,
        cutoff_frequency=cutoff_frequency,
        sampling_rate=linevoltage_VAB.fs
    )

    print(f"\nMuhammad Izhar (mi549) Yousef Zaidan (yrmz20)")
    print(f"\nWaveform signal has {linevoltage_VAB.sampling_number} samples")
    print(f"The sampling rate is {linevoltage_VAB.fs:.2f} Hz ")
    print(
        f"The dominant frequency is {linevoltage_VAB.main_frequency():.2f} Hz")

    filtered_waveform = filter_response.operate(linevoltage_VAB)

    print(
        f"\nFilter: Hamming window, fc = {filter_response.cutoff_frequency} Hz, N = {filter_response.number_of_order}")
    print(f"Main frequency peak: {filtered_waveform.main_frequency():.2f} Hz")

    analysis_report = create_analysis_report(
        before=linevoltage_VAB,
        after=filtered_waveform,
        harmonic_targets=harmonic_targets
    )

    report_path = Path(__file__).with_name("VAB_analysis_report.txt")
    analysis_report.save_to_text_file(report_path)
    print(f"\nAnalysis report saved to: {report_path.name}")

    graph_results(
        before=linevoltage_VAB,
        after=filtered_waveform,
    )
