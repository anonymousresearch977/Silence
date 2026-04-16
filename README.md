# Single-Electrode sEMG-Based Speech Decoding
## Project Overview
This project investigates the feasibility of decoding speech components 
from surface electromyography (sEMG) signals recorded from the 
perimandibular-mastoid region using a dual-channel configuration. 
Specifically, the model predicts whether a spoken word contains the 
vowel /a/ using sEMG signals captured during word production. The 
dataset includes time-domain sEMG signals and spectrogram features, 
processed for sequential modeling with a BiLSTM network.

## Dataset
The dataset is organized as follows:
- **Training Data**: `trainv2.mat` contains sEMG signals, spectrogram 
  features, word labels, and binary target labels indicating the 
  presence of the vowel /a/.
- **Test Data**: `testv2.mat` contains sEMG signals and labels for 
  evaluation on unseen words.

You can find the dataset here: [Dataset Link](https://drive.google.com/drive/folders/1DaCeynK8Co8iiimjvXe_yVnboF0PJD75?usp=drive_link)

The training and test sets have no overlapping words to ensure 
robust open-vocabulary evaluation.

## 📁 File Description
| File | Description |
|------|------------|
| `BiLSTM_v2.mlx` | Train BiLSTM and evaluate on independent test set |
| `K_Fold.mlx` | Perform 5-fold cross-validation |
| `trainv2.mat` | Training dataset (signals, spectrograms, word labels) |
| `testv2.mat` | Independent test dataset |
| `addaddVowelsMultiHot.m` | Utility function for vowel multi-hot encoding |

---

## Dataset Structure
### `trainv2.mat` contains:
- Dual-channel sEMG signals
- Spectrogram features
- Time-domain signals
- Word labels

### `testv2.mat` contains:
- sEMG signals and labels for evaluation on unseen words

To ensure smooth execution, place **all files** (scripts, functions, 
and datasets) inside the same directory.

Update the dataset loading lines in both `BiLSTM_v2.mlx` and 
`K_Fold.mlx` as shown below:

```matlab
trainData = load('trainv2.mat'); 
testData  = load('tesv2t.mat');
```

---

## Methodology
1. **Data Preprocessing**:
   - sEMG signals are segmented based on word boundaries extracted 
     from aligned audio recordings.
   - Time-frequency features are generated using Short-Time Fourier 
     Transform (STFT) with a Hamming window and 50% overlap.
   - Spectrograms from both channels are vertically concatenated to 
     form the input feature matrix (900 × T).
   - A frequency mask (0.5–450 Hz) is applied to each channel's 
     spectrogram to retain physiologically relevant content.
   - Data is normalized using a standard scaler fit on training data 
     only and applied to validation and test sets separately.
   - Gaussian noise augmentation is applied to the minority class 
     in the spectrogram domain to address class imbalance.

2. **Model Architecture**:
   - Bidirectional Long Short-Term Memory (BiLSTM) network with 
     64 hidden units captures temporal dependencies in both forward 
     and backward directions across sEMG sequences.
   - Batch normalization and dropout (p=0.5) improve generalization.
   - Two fully connected layers followed by a softmax layer perform 
     binary classification (/a/ vs. non-/a/).
   - Trained with Adam optimizer (learning rate 1×10⁻⁴), binary 
     cross-entropy loss, batch size 8, and early stopping 
     (patience=15).

3. **Evaluation**:
   - Performance metrics include Accuracy, Precision, Recall 
     (Sensitivity), Specificity, F1-Score, and AUC-ROC.
   - 5-fold cross-validation on the training set assesses robustness 
     and performance variability.
   - A permutation test (n=1,000 label shuffles) confirms statistical 
     significance of model discrimination above chance.
   - Independent test set evaluation on unseen words demonstrates 
     open-vocabulary generalization.

## Results
- The BiLSTM model achieves a test AUC of 0.611 (p=0.017, 
  permutation test, n=1,000), confirming statistically significant 
  vowel discrimination from a single sEMG electrode site.
- The BiLSTM consistently outperforms a logistic regression baseline 
  across all reported metrics, confirming the benefit of temporal 
  sequential modeling over time-averaged spectral features.
- Cross-validation and aggregated metrics reveal performance 
  variability attributable to limited dataset size, motivating 
  future data collection efforts.
