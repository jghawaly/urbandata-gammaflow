# UrbanData-GammaFlow: AI-Based Radiation Detection

A demonstration project using [GammaFlow](https://github.com/jghawaly/gammaflow) for AI-based radiation detection algorithms on the TopCoder urban radiation detection dataset.

## Overview

This project demonstrates advanced machine learning and deep learning techniques for:
- **Source Detection**: Identifying the presence of radioactive sources in time series data
- **Anomaly Detection**: Using K-Sigma, SAD, and ARAD methods
- **Explainable AI**: Saliency maps for deep learning model interpretability
- **Time Series Analysis**: Processing radiation measurements over time

## Dataset

This project uses the **TopCoder Urban Radiation Detection Challenge** dataset, which contains:
- Training and testing gamma-ray spectra in list mode format
- Multiple isotope types (Co-60, I-131, HEU, WGPu, Tc-99m)
- Time-series measurements from mobile detector in urban environments
- Ground truth labels including source times and locations

**Note**: The TopCoder dataset is not included in version control due to its size. Place it in the `topcoder/` directory with the following structure:
```
topcoder/
├── training/           # Training run CSV files
├── testing/            # Testing run CSV files
├── scorer/            # Answer key files
└── sourceInfo/        # Source template data
```

## Installation

### Prerequisites
- Python 3.8+
- GammaFlow library
- PyTorch (for ARAD detector)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/jghawaly/urbandata-gammaflow.git
cd urbandata-gammaflow
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install GammaFlow (if not already installed):
```bash
# From PyPI (when available)
pip install gammaflow

# Or from local source
pip install -e /path/to/gammaflow
```

## Project Structure

```
urbandata-gammaflow/
├── README.md
├── requirements.txt
├── examples/                    # Jupyter notebooks and training scripts
│   ├── README.md
│   ├── listmode_processing.ipynb      # List mode data processing
│   ├── k_sigma_detection.ipynb        # K-Sigma detector demo
│   ├── sad_detection.ipynb            # SAD detector demo
│   ├── arad_detection.ipynb           # ARAD detector with saliency maps
│   └── train_arad.py                  # Train ARAD autoencoder
├── src/                         # Source code
│   └── detectors/              # Detection algorithms
│       ├── __init__.py
│       ├── k_sigma.py          # K-Sigma threshold detector
│       ├── sad.py              # Spectral Anomaly Detection (PCA)
│       └── arad.py             # Autoencoder Reconstruction Anomaly Detection
├── models/                      # Trained models (not in git)
│   └── README.md
└── topcoder/                   # TopCoder dataset (not in git)
    ├── training/
    ├── testing/
    ├── scorer/
    └── sourceInfo/
```

## Detection Algorithms

### 1. K-Sigma Detector (`k_sigma.py`)
**Simple threshold-based detection with rolling background estimation**

- **Method**: Compares foreground count rate to background using statistical thresholds
- **Features**:
  - Rolling background window
  - Configurable k-sigma threshold
  - Alarm aggregation
  - Works with ROIs for improved sensitivity
- **Pros**: Fast, interpretable, no training required
- **Cons**: Limited to gross count changes

### 2. Spectral Anomaly Detection - SAD (`sad.py`)
**PCA-based multivariate anomaly detection**

- **Method**: Learns background subspace with PCA, detects deviations
- **Features**:
  - Captures spectral correlations
  - Reconstruction error as anomaly score
  - Threshold calibration by false alarm rate
  - No training labels required
- **Pros**: Multivariate, unsupervised, interpretable
- **Cons**: Linear method, may miss non-linear patterns
- **Reference**: Miller & Dubrawski (IEEE TNS, 2018)

### 3. Autoencoder Reconstruction Anomaly Detection - ARAD (`arad.py`)
**Deep learning with explainable saliency maps**

- **Method**: Convolutional autoencoder learns to reconstruct background spectra
- **Architecture**:
  - 5-layer Conv1D encoder (latent dim=8)
  - 5-layer Conv1D decoder
  - Jensen-Shannon Divergence loss
  - Batch normalization and dropout
- **Features**:
  - Non-linear pattern recognition
  - **Saliency maps** for explainability (gradient & integrated gradients)
  - GPU acceleration (CUDA/MPS)
  - Model save/load functionality
- **Pros**: Captures complex patterns, explainable AI
- **Cons**: Requires training, GPU recommended

## Usage

### Training ARAD Model

Train the autoencoder on background data:

```bash
cd examples
python train_arad.py
```

Configuration (edit at top of `train_arad.py`):
- Integration time: 5 seconds
- Energy range: 20-2900 keV, 128 bins
- Latent dimension: 8
- Training epochs: 50

Model saved to: `models/arad_background.pt`

### Running Detection Notebooks

Launch Jupyter and open any detection notebook:

```bash
cd examples
jupyter notebook
```

**Recommended order:**
1. `listmode_processing.ipynb` - Learn data format
2. `k_sigma_detection.ipynb` - Simple baseline detector
3. `sad_detection.ipynb` - PCA-based detection
4. `arad_detection.ipynb` - Deep learning with saliency maps

### Python API

```python
from src.detectors import KSigmaDetector, SADDetector, ARADDetector
from gammaflow import SpectralTimeSeries

# Load data
time_series = SpectralTimeSeries.from_list_mode(...)

# K-Sigma Detection
k_sigma = KSigmaDetector(k=3, background_window=10, foreground_window=5)
alarms = k_sigma.detect(time_series)

# SAD Detection
sad = SADDetector(n_components=5)
sad.fit(background_training)
sad.set_threshold_by_far(background_data, alarms_per_hour=0.5)
scores, alarms = sad.detect(time_series)

# ARAD Detection with Saliency
arad = ARADDetector(latent_dim=8, epochs=50)
arad.fit(background_training)
arad.load('models/arad_background.pt')
scores, alarms = arad.detect(time_series)

# Explainability
saliency = arad.compute_saliency_map(spectrum, method='gradient')
arad.plot_saliency(spectrum)  # Visual explanation
```

## Explainable AI: Saliency Maps

ARAD includes gradient-based saliency maps that show **which energy bins contribute to anomaly scores**:

- **Gradient Method**: Fast, shows ∂Loss/∂Input
- **Integrated Gradients**: Robust, integrates from baseline

**Use cases:**
- Verify model focuses on physically meaningful features
- Explain decisions to domain experts  
- Identify unexpected spectral signatures
- Debug model behavior

See `arad_detection.ipynb` for examples.

## Performance

All detectors support:
- **Threshold calibration** based on desired false alarm rate (alarms/hour)
- **Alarm aggregation** to reduce false positives
- **Event recording** with start/end times and peak metrics

## Dependencies

Core:
- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `matplotlib>=3.3.0`
- `seaborn>=0.11.0`
- `scikit-learn>=1.0.0`
- `tqdm>=4.62.0`

Deep Learning (ARAD):
- `torch>=1.10.0`

Notebooks:
- `jupyter>=1.0.0`
- `notebook>=6.4.0`

## Results

Detection performance is demonstrated in the notebooks using the TopCoder dataset. Key metrics:
- Detection rate vs false alarm rate
- Source localization accuracy
- Computational performance

## Contributing

Contributions are welcome! Areas of interest:
- Additional detection algorithms
- Performance optimizations
- New datasets
- Documentation improvements

## License

MIT License

## Acknowledgments

- **GammaFlow**: Core spectroscopy library by James Ghawaly Jr.
- **TopCoder Challenge**: Dataset from the TopCoder Urban Radiation Detection Challenge
- **SAD Method**: Based on Miller & Dubrawski (IEEE TNS, 2018)
- **Explainable AI**: Gradient-based saliency and integrated gradients

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
