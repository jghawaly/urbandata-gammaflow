"""
K-Sigma anomaly detection algorithm.

This module implements a simple but effective k-sigma detection algorithm for
identifying anomalies in gamma-ray count rate time series data.

Algorithm:
1. Maintains a rolling background window (e.g., last 60 seconds)
2. Computes mean and standard deviation of background count rate
3. Compares foreground count rate to background statistics
4. Declares alarm if (foreground - mean) / std > k threshold
5. Aggregates consecutive alarms into single detection events
6. Records alarm start/end times and peak significance
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque


@dataclass
class AlarmEvent:
    """
    Record of a detected anomaly event.
    
    Attributes
    ----------
    start_time : float
        Start time of the alarm period (seconds)
    end_time : float
        End time of the alarm period (seconds)
    peak_metric : float
        Maximum k-sigma value during the alarm period
    peak_time : float
        Time when peak metric occurred (seconds)
    duration : float
        Total duration of the alarm period (seconds)
    """
    start_time: float
    end_time: float
    peak_metric: float
    peak_time: float
    
    @property
    def duration(self) -> float:
        """Duration of the alarm period in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'peak_metric': self.peak_metric,
            'peak_time': self.peak_time,
            'duration': self.duration,
        }
    
    def __repr__(self) -> str:
        return (
            f"AlarmEvent(start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"peak={self.peak_metric:.2f}σ at {self.peak_time:.2f}s, "
            f"duration={self.duration:.2f}s)"
        )


class KSigmaDetector:
    """
    K-sigma anomaly detection for gamma-ray time series.
    
    This detector maintains a rolling background window and compares the
    foreground count rate to background statistics. An alarm is declared
    when the foreground exceeds the background by more than k standard
    deviations.
    
    Features:
    - Rolling background estimation
    - Automatic alarm aggregation (merges nearby alarms)
    - No background updates during alarm states
    - Records complete alarm events with peak metrics
    
    Parameters
    ----------
    k_threshold : float
        Number of standard deviations above background for alarm.
        Typical values: 3-5 sigma.
    background_window : float
        Duration of background window in seconds. Default is 60s.
    foreground_window : float
        Duration of foreground window in seconds. Default is 1s.
    aggregation_gap : float
        Maximum time gap between alarms to aggregate (seconds). Default is 2s.
    min_background_samples : int
        Minimum number of samples required before detection starts. Default is 10.
        
    Attributes
    ----------
    alarms : list of AlarmEvent
        List of detected alarm events
    is_alarming : bool
        Current alarm state
    current_alarm_start : float or None
        Start time of current alarm (if alarming)
        
    Examples
    --------
    >>> from gammaflow import SpectralTimeSeries
    >>> detector = KSigmaDetector(k_threshold=5.0, background_window=60.0)
    >>> 
    >>> # Process time series
    >>> detector.process_time_series(time_series)
    >>> 
    >>> # Get detected alarms
    >>> print(f"Detected {len(detector.alarms)} anomalies")
    >>> for alarm in detector.alarms:
    ...     print(alarm)
    """
    
    def __init__(
        self,
        k_threshold: float = 5.0,
        background_window: float = 60.0,
        foreground_window: float = 1.0,
        aggregation_gap: float = 2.0,
        min_background_samples: int = 10,
    ):
        self.k_threshold = k_threshold
        self.background_window = background_window
        self.foreground_window = foreground_window
        self.aggregation_gap = aggregation_gap
        self.min_background_samples = min_background_samples
        
        # State
        self.background_buffer = deque()  # (time, count_rate) tuples
        self.alarms: List[AlarmEvent] = []
        self.is_alarming = False
        self.current_alarm_start: Optional[float] = None
        self.current_alarm_peak_metric: float = -np.inf
        self.current_alarm_peak_time: Optional[float] = None
        
        # Statistics
        self.last_background_mean: Optional[float] = None
        self.last_background_std: Optional[float] = None
    
    def reset(self):
        """Reset detector state (clears buffer and alarms)."""
        self.background_buffer.clear()
        self.alarms.clear()
        self.is_alarming = False
        self.current_alarm_start = None
        self.current_alarm_peak_metric = -np.inf
        self.current_alarm_peak_time = None
        self.last_background_mean = None
        self.last_background_std = None
    
    def _update_background_buffer(self, time: float, count_rate: float):
        """Add sample to background buffer and remove old samples."""
        # Add new sample
        self.background_buffer.append((time, count_rate))
        
        # Remove samples older than background_window
        cutoff_time = time - self.background_window
        while self.background_buffer and self.background_buffer[0][0] < cutoff_time:
            self.background_buffer.popleft()
    
    def _compute_background_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute mean and standard deviation of background.
        
        Returns
        -------
        mean, std : tuple of (float or None, float or None)
            Background statistics. None if insufficient samples.
        """
        if len(self.background_buffer) < self.min_background_samples:
            return None, None
        
        count_rates = np.array([cr for _, cr in self.background_buffer])
        mean = np.mean(count_rates)
        std = np.std(count_rates, ddof=1)  # Sample std dev
        
        # Avoid division by zero
        if std < 1e-10:
            std = 1e-10
        
        return mean, std
    
    def _compute_alarm_metric(
        self,
        foreground_rate: float,
        background_mean: float,
        background_std: float,
    ) -> float:
        """
        Compute k-sigma alarm metric.
        
        Metric = (foreground - background_mean) / background_std
        
        Returns
        -------
        float
            Number of standard deviations above background
        """
        return (foreground_rate - background_mean) / background_std
    
    def _start_alarm(self, time: float, metric: float):
        """Start a new alarm period."""
        self.is_alarming = True
        self.current_alarm_start = time
        self.current_alarm_peak_metric = metric
        self.current_alarm_peak_time = time
    
    def _update_alarm_peak(self, time: float, metric: float):
        """Update peak metric if current exceeds previous peak."""
        if metric > self.current_alarm_peak_metric:
            self.current_alarm_peak_metric = metric
            self.current_alarm_peak_time = time
    
    def _end_alarm(self, time: float):
        """
        End current alarm period and record event.
        
        Checks if this alarm should be aggregated with the previous one.
        """
        if not self.is_alarming or self.current_alarm_start is None:
            return
        
        # Check if should aggregate with previous alarm
        if self.alarms:
            last_alarm = self.alarms[-1]
            time_since_last = self.current_alarm_start - last_alarm.end_time
            
            if time_since_last < self.aggregation_gap:
                # Aggregate: extend previous alarm
                last_alarm.end_time = time
                
                # Update peak if current is higher
                if self.current_alarm_peak_metric > last_alarm.peak_metric:
                    last_alarm.peak_metric = self.current_alarm_peak_metric
                    last_alarm.peak_time = self.current_alarm_peak_time
                
                # Reset current alarm state
                self.is_alarming = False
                self.current_alarm_start = None
                self.current_alarm_peak_metric = -np.inf
                self.current_alarm_peak_time = None
                return
        
        # Create new alarm event
        alarm = AlarmEvent(
            start_time=self.current_alarm_start,
            end_time=time,
            peak_metric=self.current_alarm_peak_metric,
            peak_time=self.current_alarm_peak_time,
        )
        self.alarms.append(alarm)
        
        # Reset current alarm state
        self.is_alarming = False
        self.current_alarm_start = None
        self.current_alarm_peak_metric = -np.inf
        self.current_alarm_peak_time = None
    
    def process_sample(self, time: float, count_rate: float) -> Optional[float]:
        """
        Process a single time sample.
        
        Parameters
        ----------
        time : float
            Time of measurement (seconds)
        count_rate : float
            Gross count rate (counts/second)
            
        Returns
        -------
        float or None
            Alarm metric (k-sigma) if detection is active, None otherwise
        """
        # Compute current background statistics
        bg_mean, bg_std = self._compute_background_stats()
        
        # Store for debugging/monitoring
        self.last_background_mean = bg_mean
        self.last_background_std = bg_std
        
        # Check if we have enough background for detection
        if bg_mean is None or bg_std is None:
            # Not enough background yet, buffer this sample
            self._update_background_buffer(time, count_rate)
            return None
        
        # Compute alarm metric
        alarm_metric = self._compute_alarm_metric(count_rate, bg_mean, bg_std)
        
        # Check for alarm condition
        if alarm_metric > self.k_threshold:
            # Alarm!
            if not self.is_alarming:
                # Start new alarm
                self._start_alarm(time, alarm_metric)
            else:
                # Continue alarm, update peak
                self._update_alarm_peak(time, alarm_metric)
        else:
            # No alarm
            if self.is_alarming:
                # End current alarm
                self._end_alarm(time)
            
            # Buffer this sample for future background
            self._update_background_buffer(time, count_rate)
        
        return alarm_metric
    
    def process_time_series(self, time_series) -> np.ndarray:
        """
        Process an entire time series.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process for anomaly detection
            
        Returns
        -------
        np.ndarray
            Array of alarm metrics (k-sigma values) for each time point.
            NaN where detection not yet active.
            
        Examples
        --------
        >>> detector = KSigmaDetector(k_threshold=5.0)
        >>> metrics = detector.process_time_series(time_series)
        >>> print(f"Detected {len(detector.alarms)} alarms")
        """
        # Reset state
        self.reset()
        
        # Get time points and count rates
        n_spectra = time_series.n_spectra
        
        # Get times
        if time_series.timestamps is not None:
            times = time_series.timestamps
        elif time_series.real_times is not None:
            times = np.cumsum(np.asarray(time_series.real_times, dtype=float))
        else:
            times = np.array([s.real_time for s in time_series.spectra], dtype=float)
            times = np.cumsum(times)
        
        # Get count rates
        count_rates = np.array([
            s.counts.sum() / (s.live_time if s.live_time is not None else s.real_time)
            for s in time_series.spectra
        ])
        
        # Process each sample
        metrics = np.full(n_spectra, np.nan)
        for i, (t, rate) in enumerate(zip(times, count_rates)):
            metric = self.process_sample(t, rate)
            if metric is not None:
                metrics[i] = metric
        
        # If still alarming at end, close the alarm
        if self.is_alarming:
            self._end_alarm(times[-1])
        
        return metrics
    
    def get_alarm_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of detected alarms.
        
        Returns
        -------
        dict
            Summary with keys: n_alarms, total_alarm_time, mean_duration,
            max_peak_metric, alarm_events
        """
        if not self.alarms:
            return {
                'n_alarms': 0,
                'total_alarm_time': 0.0,
                'mean_duration': 0.0,
                'max_peak_metric': 0.0,
                'alarm_events': [],
            }
        
        durations = [a.duration for a in self.alarms]
        peaks = [a.peak_metric for a in self.alarms]
        
        return {
            'n_alarms': len(self.alarms),
            'total_alarm_time': sum(durations),
            'mean_duration': np.mean(durations),
            'max_peak_metric': max(peaks),
            'alarm_events': [a.to_dict() for a in self.alarms],
        }
    
    def __repr__(self) -> str:
        return (
            f"KSigmaDetector(k={self.k_threshold}, "
            f"bg_window={self.background_window}s, "
            f"alarms={len(self.alarms)})"
        )

