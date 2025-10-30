# Examples

This directory contains example notebooks and training scripts demonstrating AI-based radiation detection techniques using GammaFlow and the TopCoder dataset.

## Notebooks

### 1. listmode_processing.ipynb
**Introduction to list mode data processing**

Demonstrates the fundamentals of working with gamma-ray list mode data:
- Loading list mode data from TopCoder dataset
- Creating `ListMode` objects from event data
- Converting to `SpectralTimeSeries` with time windowing
- Configuring integration and stride times
- Basic visualization and analysis
- Reintegration of time series data

**Topics covered:**
- List mode data format (time deltas + energies)
- Time windowing strategies
- Count rate calculations
- Gross count vs spectral analysis

---

### 2. k_sigma_detection.ipynb
**Simple threshold-based detection**

Demonstrates the K-Sigma detector for anomaly detection:
- Loading background and source data
- Configuring detector parameters (k, background window, foreground window)
- Running detection on time series
- Analyzing alarm results
- Comparing gross count vs ROI-based detection

**Key concepts:**
- Rolling background estimation
- Statistical threshold detection
- Alarm aggregation
- Region of Interest (ROI) definition for improved sensitivity

**Best for:** Fast baseline detection, interpretable results

---

### 3. sad_detection.ipynb
**Spectral Anomaly Detection using PCA**

Demonstrates the SAD detector:
- Training PCA model on background spectra
- Threshold calibration by false alarm rate (alarms/hour)
- Detection on test runs with sources
- Analyzing reconstruction error
- Comparing background vs source score distributions
- Visualizing PCA explained variance

**Key concepts:**
- Principal Component Analysis for background modeling
- Reconstruction error as anomaly metric
- Multivariate spectral analysis
- Unsupervised learning

**Best for:** Capturing spectral correlations, multivariate detection

**Reference:** Miller & Dubrawski (IEEE TNS, 2018)

---

### 4. arad_detection.ipynb  
**Deep Learning with Explainable AI**

Demonstrates the ARAD detector with saliency maps:
- Loading trained autoencoder model
- Running detection on test data
- **Computing saliency maps** for explainability
- Comparing gradient vs integrated gradients methods
- Visualizing which energy bins drive anomaly scores
- Background vs source spectrum analysis

**Key concepts:**
- Convolutional autoencoder for spectral reconstruction
- Jensen-Shannon Divergence loss
- Gradient-based saliency maps (∂Loss/∂Input)
- Integrated gradients for robust attribution
- Explainable AI for domain experts

**Best for:** Complex pattern recognition, non-linear features, explainable decisions

---

## Training Scripts

### train_arad.py
**Train the ARAD autoencoder on background data**

Trains a convolutional autoencoder to learn background spectral patterns.

**Configuration** (edit at top of file):
```python
INTEGRATION_TIME = 5.0      # seconds
STRIDE_TIME = 5.0           # seconds
LATENT_DIM = 8              # latent space dimension
EPOCHS = 50                 # training epochs
LEARNING_RATE = 0.0001      # AdamW learning rate
BATCH_SIZE = 32             # training batch size
```

**Run:**
```bash
python train_arad.py
```

**Output:**
- Trained model: `../models/arad_background.pt`
- Training history plot: `../models/arad_training_history.png`
- Reconstruction examples: `../models/arad_reconstructions.png`

**Training time:** ~5-20 minutes depending on hardware (GPU recommended)

---

## Running the Examples

### Prerequisites

1. **Install dependencies:**
```bash
cd ..
pip install -r requirements.txt
```

2. **Ensure TopCoder dataset is available:**
```
urbandata-gammaflow/topcoder/
├── training/          # *.csv list mode files
├── testing/           # *.csv list mode files
├── scorer/           # answerKey_*.csv files
└── sourceInfo/       # SourceData.csv
```

3. **For ARAD:** Install PyTorch
```bash
pip install torch
```

### Launch Jupyter

```bash
cd examples
jupyter notebook
```

### Recommended Order

1. **Start here:** `listmode_processing.ipynb`
   - Learn the data format and GammaFlow basics

2. **Baseline detector:** `k_sigma_detection.ipynb`
   - Simple, interpretable detection

3. **Multivariate detection:** `sad_detection.ipynb`
   - PCA-based spectral analysis

4. **Advanced + Explainable AI:** `arad_detection.ipynb`
   - Deep learning with saliency maps
   - **Train ARAD first** using `train_arad.py`

---

## Detection Algorithm Comparison

| Algorithm | Type | Training | Speed | Explainability | Best Use Case |
|-----------|------|----------|-------|----------------|---------------|
| **K-Sigma** | Threshold | None | Very Fast | High | Baseline, gross counts |
| **SAD** | PCA | Unsupervised | Fast | Medium | Spectral correlations |
| **ARAD** | Deep Learning | Unsupervised | Medium | **High (Saliency)** | Complex patterns, explainable AI |

---

## Key Features Demonstrated

### Data Processing
- List mode to spectral time series conversion
- Energy calibration and binning
- Time windowing strategies
- Count rate normalization

### Detection Methods
- Statistical threshold detection
- Multivariate anomaly detection
- Deep learning reconstruction error
- Explainable AI with saliency maps

### Performance Tuning
- Threshold calibration by false alarm rate
- ROI definition for improved sensitivity
- Alarm aggregation strategies
- Integration time optimization

### Visualization
- Time series count rate plots
- Spectral reconstructions
- Saliency map overlays
- Score distributions

---

## Tips for Best Results

1. **Integration Time**: 
   - Shorter (1-5s) → Better time resolution, more training samples
   - Longer (10-30s) → Better statistics, fewer samples

2. **Energy Range**:
   - ARAD requires bins divisible by 32 (e.g., 128, 256, 512)
   - Exclude very low energies (<20 keV) due to noise

3. **Threshold Calibration**:
   - Use representative background data
   - Typical: 0.1-1.0 alarms per hour
   - ANSI standards: <1 alarm per hour

4. **Saliency Maps**:
   - Use `gradient` method for speed
   - Use `integrated` method for robustness
   - Look for peaks at known isotope energies

---

## Troubleshooting

**Issue:** ARAD model file not found
- **Solution:** Run `train_arad.py` first to train the model

**Issue:** Low detection performance
- **Solution:** Try ROI-based detection (k-sigma), increase integration time, or retrain with more data

**Issue:** Too many false alarms
- **Solution:** Increase threshold, adjust `alarms_per_hour` parameter, or use alarm aggregation

**Issue:** Saliency maps look random
- **Solution:** Check that model is properly trained, verify reconstruction quality first

---

## Additional Resources

- **GammaFlow Documentation**: Core library API and usage
- **TopCoder Dataset**: Original challenge description and evaluation metrics
- **SAD Paper**: Miller & Dubrawski, IEEE TNS 2018
- **Integrated Gradients**: Sundararajan et al., ICML 2017

---

## Contributing

Have improvements or new examples? Contributions welcome!
- Additional detection algorithms
- Performance comparisons
- New visualization techniques
- Documentation improvements
