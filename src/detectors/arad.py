"""
ARAD (Autoencoder Reconstruction Anomaly Detection) for gamma-ray spectra.

Uses a convolutional autoencoder to learn background spectrum patterns
and detect anomalies via reconstruction error.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import warnings
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. ARAD detector requires PyTorch. "
        "Install with: pip install torch"
    )

from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.spectrum import Spectrum


class ARADEncoderBlock(nn.Module):
    """Encoder convolutional block with conv -> batchnorm -> maxpool -> dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.mp = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.mp(F.mish(self.bn(self.conv(x)))))


class ARADDecoderBlock(nn.Module):
    """Decoder convolutional block with upsample -> deconv -> activation -> batchnorm -> dropout."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dropout: float, is_output: bool = False):
        super().__init__()
        self.is_output = is_output
        padding = (kernel_size - 1) // 2
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding)
        
        if is_output:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Mish()
            self.bn = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.deconv(self.upsample(x))
        if self.is_output:
            return self.activation(x)
        else:
            return self.dropout(self.bn(self.activation(x)))


class ARADAutoencoder(nn.Module):
    """Convolutional autoencoder for gamma-ray spectra."""
    
    def __init__(self, n_bins: int, latent_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        
        self.n_bins = n_bins
        self.latent_dim = latent_dim
        
        # Encoder: 5 conv blocks (each divides by 2) -> 1 channel, n_bins -> 8 channels, n_bins/32
        self.encoder = nn.Sequential(
            ARADEncoderBlock(1, 8, 7, dropout),
            ARADEncoderBlock(8, 8, 5, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            nn.Flatten(),
            nn.Linear(8 * (n_bins // 32), latent_dim),
            nn.Mish(),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Decoder: linear -> 5 deconv blocks
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 8 * (n_bins // 32)),
            nn.Mish(),
            nn.BatchNorm1d(8 * (n_bins // 32)),
        )
        
        self.decoder = nn.Sequential(
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 5, dropout),
            ARADDecoderBlock(8, 1, 7, dropout, is_output=True),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        """Initialize weights with He/Xavier initialization."""
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
    
    def _normalize_input(self, x):
        """Normalize spectrum to max value (reference implementation)."""
        eps = 1e-8
        if x.dim() == 1:  # Single spectrum
            max_val = torch.max(x)
            normalized_x = x / (max_val + eps)
            normalized_x = normalized_x.view(1, 1, -1)  # Add batch & channel dims
        else:  # Batch of spectra (2D)
            max_val = torch.max(x, dim=1, keepdim=True).values
            normalized_x = x / (max_val + eps)
            normalized_x = normalized_x.view(x.size(0), 1, x.size(1))  # Add channel dim
        return normalized_x
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spectra, shape (batch, n_bins) or (n_bins,)
        
        Returns
        -------
        torch.Tensor
            Reconstructed spectra, shape (batch, n_bins) or (n_bins,), normalized to [0, 1]
        """
        # Normalize input (adds batch and channel dims)
        normalized_x = self._normalize_input(x)
        
        # Encode
        latent = self.encoder(normalized_x)
        
        # Decode
        decoded = self.decoder_linear(latent)
        decoded = decoded.view(decoded.size(0), 8, self.n_bins // 32)
        reconstructed = self.decoder(decoded)
        
        # Reshape reconstruction to match input
        if x.dim() == 1:
            reconstructed_x = reconstructed.view(-1)
        else:
            reconstructed_x = reconstructed.view(reconstructed.size(0), x.size(1))
        
        return reconstructed_x


class ARADDetector:
    """
    ARAD (Autoencoder Reconstruction Anomaly Detection) for gamma-ray spectra.
    
    Uses a convolutional autoencoder trained on background spectra to detect
    anomalies via reconstruction error (Jensen-Shannon Divergence).
    
    Parameters
    ----------
    latent_dim : int, default=8
        Dimensionality of the latent space
    dropout : float, default=0.2
        Dropout rate for regularization
    batch_size : int, default=32
        Training batch size
    learning_rate : float, default=0.01
        Initial learning rate
    epochs : int, default=50
        Maximum number of training epochs
    l1_lambda : float, default=1e-3
        L1 regularization weight
    l2_lambda : float, default=1e-3
        L2 regularization weight (via AdamW)
    early_stopping_patience : int, default=6
        Patience for early stopping
    validation_split : float, default=0.2
        Fraction of training data to use for validation
    device : str, optional
        Device to use ('cuda', 'mps', 'cpu'). If None, auto-selects.
    threshold : float, optional
        Anomaly detection threshold (JSD score)
    aggregation_gap : float, default=2.0
        Time gap (seconds) for aggregating consecutive alarms
    min_training_samples : int, default=100
        Minimum number of training samples required
    verbose : bool, default=True
        Print training progress
    
    Attributes
    ----------
    model_ : ARADAutoencoder
        Trained autoencoder model
    n_bins_ : int
        Number of energy bins
    is_fitted_ : bool
        Whether the detector has been trained
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        dropout: float = 0.2,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 50,
        l1_lambda: float = 1e-3,
        l2_lambda: float = 1e-3,
        early_stopping_patience: int = 6,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        min_training_samples: int = 100,
        verbose: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("ARAD requires PyTorch. Install with: pip install torch")
        
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.threshold = threshold
        self.aggregation_gap = aggregation_gap
        self.min_training_samples = min_training_samples
        self.verbose = verbose
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        
        if self.verbose:
            print(f"ARAD using device: {self.device}")
        
        self.model_ = None
        self.n_bins_ = None
        self.is_fitted_ = False
        self.training_history_ = {}
        
        # Detection state (for SAD-like interface)
        self.alarms: List[Dict[str, Any]] = []
    
    def _normalize_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrum to max value for loss computation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input spectrum or batch of spectra (2D: batch x bins)
        
        Returns
        -------
        torch.Tensor
            Normalized spectrum/spectra (same shape as input)
        """
        eps = 1e-8
        if x.dim() == 1:
            max_val = torch.max(x)
            return x / (max_val + eps)
        else:
            max_val = torch.max(x, dim=1, keepdim=True).values
            return x / (max_val + eps)
    
    def _jsd_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Jensen-Shannon Divergence loss.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True (normalized) spectra
        y_pred : torch.Tensor
            Predicted (normalized) spectra
        
        Returns
        -------
        torch.Tensor
            Mean JSD across batch
        """
        # Clamp for numerical stability
        y_true = torch.clamp(y_true, min=1e-10, max=1.0)
        y_pred = torch.clamp(y_pred, min=1e-10, max=1.0)
        
        m = 0.5 * (y_true + y_pred)
        kld_pm = torch.sum(y_true * torch.log(y_true / m), dim=-1)
        kld_qm = torch.sum(y_pred * torch.log(y_pred / m), dim=-1)
        
        return torch.sqrt(0.5 * (kld_pm + kld_qm)).mean()
    
    def _compute_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss (JSD + L1 regularization).
        
        Parameters
        ----------
        y_true : torch.Tensor
            True spectra (unnormalized) - will be normalized by model
        y_pred : torch.Tensor
            Predicted spectra (normalized by model)
        
        Returns
        -------
        torch.Tensor
            Total loss
        """
        # Model now normalizes internally, so y_pred is already normalized
        # We need to normalize y_true the same way for comparison
        y_true_norm = self._normalize_spectrum(y_true)
        
        # Reconstruction loss  
        recon_loss = self._jsd_loss(y_true_norm, y_pred)
        
        # L1 regularization
        l1_norm = sum(param.abs().sum() for param in self.model_.parameters())
        
        return recon_loss + self.l1_lambda * l1_norm
    
    def fit(
        self,
        background_training: SpectralTimeSeries,
        validation_data: Optional[SpectralTimeSeries] = None
    ) -> 'ARADDetector':
        """
        Train the ARAD detector on background spectra.
        
        Parameters
        ----------
        background_training : SpectralTimeSeries
            Background spectra for training
        validation_data : SpectralTimeSeries, optional
            Validation data. If None, uses validation_split of training data.
        
        Returns
        -------
        self
            Fitted detector
        """
        # Extract count rate data (counts normalized by time)
        counts = background_training.counts
        times = background_training.live_times
        # Check if times is unusable (None, object array with None, or contains NaN)
        if times is None or times.dtype == object or (times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = background_training.real_times
        
        # Compute count rates
        training_spectra = counts / times[:, np.newaxis]
        
        if training_spectra.shape[0] < self.min_training_samples:
            raise ValueError(
                f"Need at least {self.min_training_samples} training samples, "
                f"got {training_spectra.shape[0]}"
            )
        
        self.n_bins_ = training_spectra.shape[1]
        
        # Check that n_bins is divisible by 32 (5 pooling layers)
        if self.n_bins_ % 32 != 0:
            raise ValueError(
                f"Number of bins ({self.n_bins_}) must be divisible by 32 "
                f"for 5 pooling layers. Consider rebinning."
            )
        
        # Split into train/validation if needed
        if validation_data is None:
            n_train = int(len(training_spectra) * (1 - self.validation_split))
            indices = np.random.permutation(len(training_spectra))
            train_data = training_spectra[indices[:n_train]]
            val_data = training_spectra[indices[n_train:]]
        else:
            train_data = training_spectra
            # Extract count rates from validation data
            val_counts = validation_data.counts
            val_times = validation_data.live_times
            # Check if times is unusable (None, object array with None, or contains NaN)
            if val_times is None or val_times.dtype == object or (val_times.dtype in [np.float32, np.float64] and np.any(np.isnan(val_times))):
                val_times = validation_data.real_times
            val_data = val_counts / val_times[:, np.newaxis]
        
        if self.verbose:
            print(f"Training on {len(train_data)} spectra, validating on {len(val_data)}")
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(train_data).to(self.device)
        val_tensor = torch.FloatTensor(val_data).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        self.model_ = ARADAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
            eps=0.1
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            
            for batch, in train_loader:
                optimizer.zero_grad()
                reconstructed = self.model_(batch)
                loss = self._compute_loss(batch, reconstructed)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model_.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch, in val_loader:
                    reconstructed = self.model_(batch)
                    loss = self._compute_loss(batch, reconstructed)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss - 1e-4:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.training_history_ = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        
        return self
    
    def score_spectrum(self, spectrum: Spectrum) -> float:
        """
        Score a single spectrum (reconstruction error).
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to score
        
        Returns
        -------
        float
            Anomaly score (JSD between input and reconstruction)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Extract count rate
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        
        if len(spectrum_data) != self.n_bins_:
            raise ValueError(
                f"Spectrum has {len(spectrum_data)} bins, expected {self.n_bins_}"
            )
        
        # Convert to tensor
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        
        # Get reconstruction
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)
        
        # Normalize both
        x_norm = self._normalize_spectrum(x)
        reconstructed_norm = self._normalize_spectrum(reconstructed)
        
        # Compute JSD
        score = self._jsd_loss(x_norm, reconstructed_norm).item()
        
        return score
    
    def detect(
        self,
        time_series: SpectralTimeSeries
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect anomalies in a time series.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to analyze
        
        Returns
        -------
        scores : np.ndarray
            Anomaly scores for each spectrum
        alarms : List[Dict[str, Any]]
            List of alarm events with keys:
            - 'start_time': Alarm start time
            - 'end_time': Alarm end time
            - 'peak_score': Maximum score during alarm
            - 'peak_time': Time of maximum score
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before detection")
        
        if self.threshold is None:
            raise ValueError(
                "Threshold must be set before detection. "
                "Use set_threshold_by_far() or set threshold manually."
            )
        
        # Score all spectra
        scores = np.zeros(time_series.n_spectra)
        for i in range(time_series.n_spectra):
            scores[i] = self.score_spectrum(time_series[i])
        
        # Detect alarms
        alarms = []
        in_alarm = False
        alarm_start = None
        alarm_scores = []
        alarm_times = []
        
        timestamps = time_series.timestamps
        
        for i, (score, t) in enumerate(zip(scores, timestamps)):
            if score > self.threshold:
                if not in_alarm:
                    # Start new alarm
                    in_alarm = True
                    alarm_start = t
                    alarm_scores = [score]
                    alarm_times = [t]
                else:
                    # Continue alarm
                    alarm_scores.append(score)
                    alarm_times.append(t)
            else:
                if in_alarm:
                    # Check if we should end alarm or extend it
                    if i < len(timestamps) - 1 and (timestamps[i + 1] - alarm_times[-1]) < self.aggregation_gap:
                        # Extend alarm (gap tolerance)
                        alarm_scores.append(score)
                        alarm_times.append(t)
                    else:
                        # End alarm
                        peak_idx = np.argmax(alarm_scores)
                        alarms.append({
                            'start_time': alarm_start,
                            'end_time': alarm_times[-1],
                            'peak_score': alarm_scores[peak_idx],
                            'peak_time': alarm_times[peak_idx]
                        })
                        in_alarm = False
        
        # Handle alarm at end of series
        if in_alarm:
            peak_idx = np.argmax(alarm_scores)
            alarms.append({
                'start_time': alarm_start,
                'end_time': alarm_times[-1],
                'peak_score': alarm_scores[peak_idx],
                'peak_time': alarm_times[peak_idx]
            })
        
        return scores, alarms
    
    def process_time_series(self, time_series: SpectralTimeSeries) -> np.ndarray:
        """
        Process an entire time series for anomaly detection.
        
        This method is provided for API compatibility with SAD detector.
        It calls detect() and stores the alarms in self.alarms.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process for anomaly detection
            
        Returns
        -------
        np.ndarray
            Array of ARAD scores for each time point
            
        Raises
        ------
        RuntimeError
            If detector not trained or threshold not set
            
        Examples
        --------
        >>> detector = ARADDetector()
        >>> detector.fit(background_data)
        >>> detector.set_threshold_by_far(background_data, alarms_per_hour=0.5)
        >>> 
        >>> scores = detector.process_time_series(test_data)
        >>> print(f"Detected {len(detector.alarms)} anomalies")
        >>> for alarm in detector.alarms:
        ...     print(f"  {alarm}")
        """
        scores, alarms = self.detect(time_series)
        self.alarms = alarms
        return scores
    
    def set_threshold_by_far(
        self,
        background_data: SpectralTimeSeries,
        alarms_per_hour: float,
        max_iterations: int = 20
    ) -> float:
        """
        Set detection threshold based on desired false alarm rate.
        
        This method iteratively adjusts the threshold to achieve the target
        false alarm rate, expressed as alarms per hour. This is the standard
        metric for operational radiation detection systems (ANSI N42.48).
        
        The method uses binary search with actual alarm processing to account
        for alarm aggregation (consecutive high-scoring spectra that merge into
        a single alarm event).
        
        Parameters
        ----------
        background_data : SpectralTimeSeries
            Background data to calibrate threshold
        alarms_per_hour : float
            Target false alarm rate (alarms per hour)
        max_iterations : int, optional
            Maximum number of binary search iterations (default: 20)
        
        Returns
        -------
        float
            Calibrated threshold
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before setting threshold")
        
        # Score all background spectra
        scores = np.array([
            self.score_spectrum(background_data[i])
            for i in range(background_data.n_spectra)
        ])
        
        # Calculate total observation time from real_times (actual counting time)
        total_time_seconds = np.sum(background_data.real_times)
        total_time_hours = total_time_seconds / 3600.0
        
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
        
        if self.verbose:
            print(f"\nCalibrating threshold for {alarms_per_hour:.2f} alarms/hour...")
            print(f"  Background data: {len(scores)} spectra over {total_time_hours:.2f} hours")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Score mean ± std: {scores.mean():.4f} ± {scores.std():.4f}")
            print(f"  Starting binary search...")
        
        for iteration in range(max_iterations):
            # Try this threshold
            test_threshold = (low_threshold + high_threshold) / 2
            self.threshold = test_threshold
            
            # Process to count alarms (this resets and populates self.alarms)
            _ = self.process_time_series(background_data)
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
            
            if self.verbose:
                print(f"    Iter {iteration+1}: threshold={test_threshold:.6f} → {n_alarms} alarms ({observed_far:.2f}/hr)")
            
            # Adjust search range
            if observed_far > alarms_per_hour:
                # Too many alarms, increase threshold
                low_threshold = test_threshold
            else:
                # Too few alarms, decrease threshold
                high_threshold = test_threshold
            
            # Check convergence
            if far_diff < 0.1 * alarms_per_hour or (high_threshold - low_threshold) < 1e-8:
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations")
                break
        
        # Set the best threshold found
        self.threshold = best_threshold
        
        # Final pass to get alarm count with best threshold
        self.process_time_series(background_data)
        final_far = len(self.alarms) / total_time_hours
        
        if self.verbose:
            print(f"\n✅ Threshold set: {self.threshold:.6f}")
            print(f"   Achieved FAR: {final_far:.2f} alarms/hour ({len(self.alarms)} alarms)")
            print(f"   Target FAR: {alarms_per_hour:.2f} alarms/hour")
        
        return self.threshold
    
    def save(self, path: str):
        """
        Save trained model to file.
        
        Parameters
        ----------
        path : str
            Path to save model
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dict = {
            'model_state': self.model_.state_dict(),
            'n_bins': self.n_bins_,
            'latent_dim': self.latent_dim,
            'dropout': self.dropout,
            'threshold': self.threshold,
            'training_history': self.training_history_
        }
        
        torch.save(save_dict, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load trained model from file.
        
        Parameters
        ----------
        path : str
            Path to saved model
        """
        save_dict = torch.load(path, map_location=self.device)
        
        self.n_bins_ = save_dict['n_bins']
        self.latent_dim = save_dict['latent_dim']
        self.dropout = save_dict['dropout']
        self.threshold = save_dict['threshold']
        self.training_history_ = save_dict.get('training_history', {})
        
        self.model_ = ARADAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            dropout=self.dropout
        ).to(self.device)
        
        self.model_.load_state_dict(save_dict['model_state'])
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.
        
        Returns
        -------
        dict
            Training and validation loss history
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")
        
        return self.training_history_
    
    def reconstruct(self, spectrum: Spectrum) -> np.ndarray:
        """
        Reconstruct a spectrum through the autoencoder.
        
        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum
        
        Returns
        -------
        np.ndarray
            Reconstructed spectrum (count rates)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before reconstruction")
        
        # Extract count rate
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)
        
        # Denormalize to match input scale
        max_val = np.max(spectrum_data)
        reconstructed_np = reconstructed.cpu().numpy().flatten()
        
        return reconstructed_np * max_val
    
    def compute_saliency_map(self, spectrum: Spectrum, method: str = 'gradient') -> np.ndarray:
        """
        Compute saliency map showing which energy bins contribute most to anomaly score.
        
        This provides explainability by highlighting spectral features that the
        autoencoder struggles to reconstruct.
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to analyze
        method : str, default='gradient'
            Saliency computation method:
            - 'gradient': Simple gradient-based attribution (∂Loss/∂Input)
            - 'integrated': Integrated gradients (more robust)
        
        Returns
        -------
        np.ndarray
            Saliency map (same length as spectrum). Higher values indicate
            energy bins that contribute more to the anomaly score.
        
        Examples
        --------
        >>> saliency = detector.compute_saliency_map(anomalous_spectrum)
        >>> plt.plot(spectrum.energy_centers, saliency)
        >>> plt.xlabel('Energy (keV)')
        >>> plt.ylabel('Saliency')
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before computing saliency")
        
        if method == 'gradient':
            return self._compute_gradient_saliency(spectrum)
        elif method == 'integrated':
            return self._compute_integrated_gradients(spectrum)
        else:
            raise ValueError(f"Unknown saliency method: {method}")
    
    def _compute_gradient_saliency(self, spectrum: Spectrum) -> np.ndarray:
        """
        Compute gradient-based saliency: |∂Loss/∂Input|.
        
        Shows which input features (energy bins) have the largest effect
        on the reconstruction error.
        """
        # Extract count rate
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        
        # Create tensor with gradient tracking
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        x.requires_grad = True
        
        # Forward pass
        self.model_.eval()
        reconstructed = self.model_(x)
        
        # Compute loss
        x_norm = self._normalize_spectrum(x)
        loss = self._jsd_loss(x_norm, reconstructed)
        
        # Backpropagate
        loss.backward()
        
        # Get absolute gradients
        saliency = np.abs(x.grad.cpu().numpy().squeeze())
        
        return saliency
    
    def _compute_integrated_gradients(
        self, 
        spectrum: Spectrum, 
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients for more robust attribution.
        
        Integrates gradients along a path from baseline to input, providing
        more stable and interpretable attributions than simple gradients.
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to analyze
        n_steps : int
            Number of integration steps
        baseline : np.ndarray, optional
            Baseline spectrum (default: zeros)
        """
        # Extract count rate
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        
        # Set baseline (default to zeros)
        if baseline is None:
            baseline = np.zeros_like(spectrum_data)
        
        # Convert to tensors
        x = torch.FloatTensor(spectrum_data).unsqueeze(0).to(self.device)
        baseline_tensor = torch.FloatTensor(baseline).unsqueeze(0).to(self.device)
        
        # Accumulate gradients along path
        integrated_grads = np.zeros_like(spectrum_data)
        
        for i in range(n_steps):
            # Interpolate between baseline and input
            alpha = (i + 1) / n_steps
            interpolated = baseline_tensor + alpha * (x - baseline_tensor)
            interpolated.requires_grad = True
            
            # Forward pass
            self.model_.eval()
            reconstructed = self.model_(interpolated)
            
            # Compute loss
            interp_norm = self._normalize_spectrum(interpolated)
            loss = self._jsd_loss(interp_norm, reconstructed)
            
            # Backpropagate
            loss.backward()
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.cpu().numpy().squeeze()
        
        # Average gradients and multiply by (input - baseline)
        integrated_grads = integrated_grads / n_steps
        integrated_grads = integrated_grads * (spectrum_data - baseline)
        
        return np.abs(integrated_grads)
    
    def plot_saliency(
        self,
        spectrum: Spectrum,
        method: str = 'gradient',
        figsize: tuple = (14, 8),
        show_reconstruction: bool = True
    ):
        """
        Plot spectrum with saliency map overlay for explainability.
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to visualize
        method : str, default='gradient'
            Saliency method ('gradient' or 'integrated')
        figsize : tuple, default=(14, 8)
            Figure size
        show_reconstruction : bool, default=True
            Whether to show reconstruction comparison
        
        Returns
        -------
        fig, axes
            Matplotlib figure and axes objects
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            raise ImportError("Matplotlib required for plotting. Install with: pip install matplotlib")
        
        # Get data
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        count_rate = counts / time
        energy_centers = spectrum.energy_centers
        
        # Compute saliency and reconstruction
        saliency = self.compute_saliency_map(spectrum, method=method)
        score = self.score_spectrum(spectrum)
        
        # Create figure
        if show_reconstruction:
            fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
            reconstructed = self.reconstruct(spectrum)
        else:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            axes = [ax]
        
        # Plot 1: Spectrum with saliency overlay
        ax1 = axes[0]
        
        # Plot spectrum
        ax1.plot(energy_centers, count_rate, 'k-', linewidth=2, label='Original Spectrum', zorder=2)
        
        # Overlay saliency as colored background
        # Normalize saliency for colormap
        saliency_norm = saliency / (np.max(saliency) + 1e-10)
        
        # Create custom colormap (white to red)
        cmap = LinearSegmentedColormap.from_list('saliency', ['white', 'yellow', 'orange', 'red'])
        
        # Plot saliency as filled area
        for i in range(len(energy_centers) - 1):
            ax1.axvspan(
                energy_centers[i], 
                energy_centers[i+1], 
                alpha=0.3 * saliency_norm[i], 
                color='red',
                zorder=1
            )
        
        ax1.set_yscale('log')
        ax1.set_ylabel('Count Rate (s$^{-1}$)', fontsize=12)
        ax1.set_title(
            f'Spectrum with Saliency Overlay (Score={score:.4f}, Method={method})',
            fontsize=13,
            fontweight='bold'
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, zorder=0)
        
        # Plot 2: Reconstruction comparison (if requested)
        if show_reconstruction:
            ax2 = axes[1]
            ax2.plot(energy_centers, count_rate, 'k-', linewidth=1.5, label='Original', alpha=0.7)
            ax2.plot(energy_centers, reconstructed, 'r-', linewidth=1.5, label='Reconstructed', alpha=0.7)
            
            ax2.set_yscale('log')
            ax2.set_xlabel('Energy (keV)', fontsize=12)
            ax2.set_ylabel('Count Rate (s$^{-1}$)', fontsize=12)
            ax2.set_title('Reconstruction Comparison', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Energy (keV)', fontsize=12)
        
        plt.tight_layout()
        
        return fig, axes

