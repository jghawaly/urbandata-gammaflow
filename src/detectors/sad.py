"""
Spectral Anomaly Detection (SAD) using PCA-based reconstruction error.

This module implements the SAD algorithm which learns a low-dimensional subspace
of background spectra using PCA, then detects anomalies by measuring how poorly
new spectra can be reconstructed using only that subspace.

References
----------
Miller, K., & Dubrawski, A. (2018). Gamma-ray source detection with small sensors.
    IEEE Transactions on Nuclear Science, 65(4), 1047-1058.

"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.decomposition import PCA


@dataclass
class AlarmEvent:
    """
    Represents a detected anomaly event.
    
    Attributes
    ----------
    start_time : float
        Start time of alarm (seconds)
    end_time : float
        End time of alarm (seconds)
    peak_metric : float
        Peak SAD score during alarm
    peak_time : float
        Time of peak SAD score
    duration : float
        Duration of alarm in seconds
    """
    start_time: float
    end_time: float
    peak_metric: float
    peak_time: float
    
    @property
    def duration(self) -> float:
        """Duration of alarm in seconds."""
        return self.end_time - self.start_time
    
    def __repr__(self) -> str:
        return (f"AlarmEvent(start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
                f"peak={self.peak_metric:.2f} at {self.peak_time:.2f}s, "
                f"duration={self.duration:.2f}s)")


class SADDetector:
    """
    Spectral Anomaly Detector using PCA-based reconstruction error.
    
    This detector learns a low-dimensional subspace of background spectra using
    Principal Component Analysis (PCA). New spectra are scored by measuring their
    reconstruction error when projected onto this subspace:
    
        SAD(x) = ||(I - UU^T)x||^2
    
    where U ∈ R^(n×k) is the orthonormal basis of the learned subspace.
    
    Parameters
    ----------
    n_components : int, optional
        Number of principal components to retain (default: 5)
    threshold : float or None, optional
        SAD score threshold for declaring anomalies. If None, must be set
        later using set_threshold() or set_threshold_by_far() (default: None)
    normalize : bool, optional
        If True, normalize spectra to unit integral before PCA (default: True)
    aggregation_gap : float, optional
        Time gap (seconds) for aggregating consecutive alarms (default: 2.0)
    min_training_samples : int, optional
        Minimum number of background samples needed for training (default: 20)
    
    Attributes
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA model
    is_trained : bool
        Whether the detector has been trained
    alarms : list of AlarmEvent
        Detected alarm events
    
    Examples
    --------
    >>> # Train on background data
    >>> detector = SADDetector(n_components=5)
    >>> detector.fit(background_time_series)
    >>> 
    >>> # Set threshold based on 1% false alarm rate
    >>> detector.set_threshold_by_far(background_time_series, false_alarm_rate=0.01)
    >>> 
    >>> # Process new data
    >>> scores = detector.process_time_series(test_time_series)
    >>> print(f"Detected {len(detector.alarms)} anomalies")
    """
    
    def __init__(
        self,
        n_components: int = 5,
        threshold: Optional[float] = None,
        normalize: bool = True,
        aggregation_gap: float = 2.0,
        min_training_samples: int = 20
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.normalize = normalize
        self.aggregation_gap = aggregation_gap
        self.min_training_samples = min_training_samples
        
        # PCA model
        self.pca: Optional[PCA] = None
        self.is_trained = False
        
        # For normalization
        self.mean_spectrum: Optional[np.ndarray] = None
        
        # Detection state
        self.alarms: List[AlarmEvent] = []
        self.is_alarming = False
        self.current_alarm_start: Optional[float] = None
        self.current_alarm_peak_metric: float = -np.inf
        self.current_alarm_peak_time: Optional[float] = None
    
    def _normalize_spectrum(self, counts: np.ndarray) -> np.ndarray:
        """
        Normalize spectrum to unit integral.
        
        Parameters
        ----------
        counts : np.ndarray
            Raw spectrum counts
            
        Returns
        -------
        np.ndarray
            Normalized spectrum (sums to 1)
        """
        total = counts.sum()
        if total > 0:
            return counts / total
        return counts
    
    def _prepare_spectrum(self, counts: np.ndarray) -> np.ndarray:
        """
        Prepare spectrum for PCA (normalization if enabled).
        
        Parameters
        ----------
        counts : np.ndarray
            Raw spectrum counts
            
        Returns
        -------
        np.ndarray
            Prepared spectrum
        """
        if self.normalize:
            return self._normalize_spectrum(counts)
        return counts
    
    def fit(self, background_data) -> 'SADDetector':
        """
        Train the detector on background (source-absent) data.
        
        Parameters
        ----------
        background_data : SpectralTimeSeries or Spectra
            Background spectra for training. Can be a SpectralTimeSeries
            or Spectra object from GammaFlow.
            
        Returns
        -------
        self
            Returns self for method chaining
            
        Raises
        ------
        ValueError
            If insufficient training data provided
            
        Examples
        --------
        >>> detector = SADDetector(n_components=10)
        >>> detector.fit(background_time_series)
        >>> print(f"Trained on {detector.pca.n_samples_} spectra")
        """
        # Extract spectra
        if hasattr(background_data, 'spectra'):
            spectra = background_data.spectra
        else:
            raise ValueError("background_data must have a 'spectra' attribute")
        
        n_samples = len(spectra)
        if n_samples < self.min_training_samples:
            raise ValueError(
                f"Need at least {self.min_training_samples} training samples, "
                f"got {n_samples}"
            )
        
        # Extract and prepare spectra
        n_bins = spectra[0].counts.shape[0]
        X = np.zeros((n_samples, n_bins))
        
        for i, spec in enumerate(spectra):
            X[i, :] = self._prepare_spectrum(spec.counts)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        self.is_trained = True
        
        return self
    
    def score_spectrum(self, spectrum) -> float:
        """
        Compute SAD score for a single spectrum.
        
        The SAD score is the squared norm of the reconstruction error when
        projecting the spectrum onto the learned background subspace:
        
            SAD(x) = ||(I - UU^T)x||^2
        
        Parameters
        ----------
        spectrum : Spectrum or np.ndarray
            Spectrum to score. Can be a GammaFlow Spectrum object or
            a numpy array of counts.
            
        Returns
        -------
        float
            SAD score (reconstruction error)
            
        Raises
        ------
        RuntimeError
            If detector has not been trained
            
        Examples
        --------
        >>> score = detector.score_spectrum(test_spectrum)
        >>> if score > detector.threshold:
        ...     print("Anomaly detected!")
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before scoring. Call fit() first.")
        
        # Extract counts
        if hasattr(spectrum, 'counts'):
            counts = spectrum.counts
        else:
            counts = spectrum
        
        # Prepare spectrum
        x = self._prepare_spectrum(counts).reshape(1, -1)
        
        # Project onto subspace and back
        x_reconstructed = self.pca.inverse_transform(self.pca.transform(x))
        
        # Compute reconstruction error
        residual = x - x_reconstructed
        sad_score = np.sum(residual ** 2)
        
        return float(sad_score)
    
    def score_time_series(self, time_series) -> np.ndarray:
        """
        Compute SAD scores for all spectra in a time series.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to score
            
        Returns
        -------
        np.ndarray
            Array of SAD scores, one per spectrum
            
        Examples
        --------
        >>> scores = detector.score_time_series(test_data)
        >>> plt.plot(test_data.timestamps, scores)
        """
        scores = np.array([
            self.score_spectrum(spec) for spec in time_series.spectra
        ])
        return scores
    
    def set_threshold(self, threshold: float) -> 'SADDetector':
        """
        Manually set the detection threshold.
        
        Parameters
        ----------
        threshold : float
            SAD score threshold for declaring anomalies
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        self.threshold = threshold
        return self
    
    def set_threshold_by_far(
        self,
        background_data,
        alarms_per_hour: float = 1.0,
        max_iterations: int = 20
    ) -> 'SADDetector':
        """
        Set threshold based on desired false alarm rate (alarms per hour).
        
        This method iteratively adjusts the threshold to achieve the target
        false alarm rate, expressed as alarms per hour. This is the standard
        metric for operational radiation detection systems (ANSI N42.48).
        
        Parameters
        ----------
        background_data : SpectralTimeSeries or Spectra
            Background data for threshold calibration. Must be a time series
            with temporal information to calculate alarms per hour.
        alarms_per_hour : float, optional
            Desired false alarm rate in alarms per hour (default: 1.0).
            ANSI standards typically require < 1 alarm/hour.
        max_iterations : int, optional
            Maximum number of iterations for threshold search (default: 20)
            
        Returns
        -------
        self
            Returns self for method chaining
            
        Examples
        --------
        >>> # Set threshold for 0.5 alarms per hour (ANSI compliant)
        >>> detector.fit(train_background)
        >>> detector.set_threshold_by_far(val_background, alarms_per_hour=0.5)
        >>> print(f"Threshold: {detector.threshold:.6f}")
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before setting threshold. Call fit() first.")
        
        # Score all background spectra
        scores = self.score_time_series(background_data)
        
        # Get total observation time in hours
        # Use sum of real_times (actual counting time) rather than timestamp span
        # This is more accurate, especially when data has gaps or artificial offsets
        if hasattr(background_data, 'real_times') and background_data.real_times is not None:
            total_time_seconds = np.sum(background_data.real_times)
            total_time_hours = total_time_seconds / 3600.0
        elif hasattr(background_data, 'spectra'):
            # Fallback: extract from individual spectra
            total_time_seconds = sum(s.real_time for s in background_data.spectra)
            total_time_hours = total_time_seconds / 3600.0
        else:
            raise ValueError("Cannot determine observation time from background_data")
        
        if total_time_hours <= 0:
            raise ValueError(f"Invalid observation time: {total_time_hours} hours")
        
        # Binary search for the right threshold
        # Start with percentile-based initial guess
        initial_percentile = max(0.1, min(99.9, 100 * (1 - alarms_per_hour / (60 * len(scores)))))
        low_threshold = np.min(scores)
        high_threshold = np.max(scores) * 1.5
        
        best_threshold = float(np.percentile(scores, initial_percentile))
        best_far_diff = float('inf')
        best_observed_far = 0.0
        
        for iteration in range(max_iterations):
            # Try this threshold
            test_threshold = (low_threshold + high_threshold) / 2
            self.threshold = test_threshold
            
            # Process to count alarms
            temp_scores = self.process_time_series(background_data)
            n_alarms = len(self.alarms)
            observed_far = n_alarms / total_time_hours
            
            # Check if this is the best so far
            # Prefer lower thresholds (more sensitive) when we can't hit target exactly
            far_diff = abs(observed_far - alarms_per_hour)
            is_better = False
            
            if far_diff < best_far_diff:
                # Closer to target
                is_better = True
            elif far_diff == best_far_diff:
                # Same distance from target
                if observed_far > best_observed_far:
                    # Higher FAR (more sensitive) - prefer this
                    is_better = True
                elif observed_far == best_observed_far:
                    # Same FAR - prefer lower threshold (more sensitive)
                    is_better = test_threshold < best_threshold
            
            if is_better:
                best_far_diff = far_diff
                best_threshold = test_threshold
                best_observed_far = observed_far
            
            # Adjust search range
            if observed_far > alarms_per_hour:
                # Too many alarms, increase threshold
                low_threshold = test_threshold
            else:
                # Too few alarms, decrease threshold
                high_threshold = test_threshold
            
            # Check convergence
            if far_diff < 0.1 * alarms_per_hour or (high_threshold - low_threshold) < 1e-8:
                break
        
        # Set the best threshold found
        self.threshold = best_threshold
        
        # Final pass to get alarm count with best threshold
        self.process_time_series(background_data)
        final_far = len(self.alarms) / total_time_hours
        
        return self
    
    def reset(self):
        """Reset detection state (alarms and current alarm tracking)."""
        self.alarms = []
        self.is_alarming = False
        self.current_alarm_start = None
        self.current_alarm_peak_metric = -np.inf
        self.current_alarm_peak_time = None
    
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
    
    def process_time_series(self, time_series) -> np.ndarray:
        """
        Process an entire time series for anomaly detection.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process for anomaly detection
            
        Returns
        -------
        np.ndarray
            Array of SAD scores for each time point
            
        Raises
        ------
        RuntimeError
            If detector not trained or threshold not set
            
        Examples
        --------
        >>> detector = SADDetector(n_components=5)
        >>> detector.fit(background_data)
        >>> detector.set_threshold_by_far(background_data, false_alarm_rate=0.01)
        >>> 
        >>> scores = detector.process_time_series(test_data)
        >>> print(f"Detected {len(detector.alarms)} anomalies")
        >>> for alarm in detector.alarms:
        ...     print(f"  {alarm}")
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before processing. Call fit() first.")
        
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Call set_threshold() or set_threshold_by_far() first."
            )
        
        # Reset state
        self.reset()
        
        # Get times
        if time_series.timestamps is not None:
            times = time_series.timestamps
        elif time_series.real_times is not None:
            times = np.cumsum(np.asarray(time_series.real_times, dtype=float))
        else:
            times = np.array([s.real_time for s in time_series.spectra], dtype=float)
            times = np.cumsum(times)
        
        # Score all spectra
        scores = self.score_time_series(time_series)
        
        # Process each sample for alarm detection
        for i, (time, score) in enumerate(zip(times, scores)):
            if score > self.threshold:
                # Anomaly detected
                if not self.is_alarming:
                    self._start_alarm(time, score)
                else:
                    self._update_alarm_peak(time, score)
            else:
                # No anomaly
                if self.is_alarming:
                    self._end_alarm(time)
        
        # Close any open alarm at end
        if self.is_alarming:
            self._end_alarm(times[-1])
        
        return scores
    
    def get_alarm_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of detected alarms.
        
        Returns
        -------
        dict
            Dictionary with alarm statistics:
            - n_alarms: number of alarms
            - total_alarm_time: total time in alarm state
            - mean_duration: mean alarm duration
            - max_peak_metric: highest SAD score across all alarms
        """
        if not self.alarms:
            return {
                'n_alarms': 0,
                'total_alarm_time': 0.0,
                'mean_duration': 0.0,
                'max_peak_metric': 0.0
            }
        
        return {
            'n_alarms': len(self.alarms),
            'total_alarm_time': sum(a.duration for a in self.alarms),
            'mean_duration': np.mean([a.duration for a in self.alarms]),
            'max_peak_metric': max(a.peak_metric for a in self.alarms)
        }
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Get the explained variance ratio for each principal component.
        
        Returns
        -------
        np.ndarray
            Array of explained variance ratios
            
        Raises
        ------
        RuntimeError
            If detector has not been trained
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained first. Call fit().")
        
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance_explained(self) -> float:
        """
        Get total variance explained by all retained components.
        
        Returns
        -------
        float
            Cumulative explained variance (0-1)
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained first. Call fit().")
        
        return float(self.pca.explained_variance_ratio_.sum())

