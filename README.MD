# Machine Learning Project

## Team Members

- Tomás Fonseca, ist66325
- Tiago Simões, ist96329

## Project Description

This repository contains implementations for four Machine Learning problems covering both classification and regression tasks on image and tabular data.

**Classification Problems:**

- **Class1**: Binary image classification (Nevi vs Melanoma) using 28x28 RGB images. Implements Logistic Regression, Naive Bayes, SVM, and CNN models with various class balancing techniques (SMOTE, oversampling, undersampling, data augmentation).
- **Class2**: 6-class image classification (Nevi, Melanoma, Vascular Lesions, Granulocytes, Basophils, Lymphocytes) using 28x28 RGB images. Implements a CNN model with class balancing.

**Regression Problems:**

- **Reg1**: Linear regression with regularization (Ridge, Lasso, ElasticNet) using cross-validation and hyperparameter tuning via grid search.
- **Reg2**: Robust regression using RANSAC clustering to handle outliers, with two-cluster modeling.

## Repository Structure

```
Project_ML/
├── Class1/ # Binary image classification (Nevi vs Melanoma)
│ ├── g04_class1.py # Main script: Logistic Regression, NB, SVM, CNN
│ ├── Xtrain_Classification1.npy # Training images (28x28x3 RGB)
│ ├── ytrain_Classification1.npy # Training labels (binary)
│ ├── Xtest_Classification1.npy # Test images
│ └── ytest_Classification1.npy # Generated predictions
├── Class2/ # 6-class image classification
│ ├── g04_class2.py # Main script: CNN with class balancing
│ ├── Xtrain_Classification2.npy # Training images (28x28x3 RGB)
│ ├── ytrain_Classification2.npy # Training labels (6 classes)
│ ├── Xtest_Classification2.npy # Test images
│ └── ytest_Classification2.npy # Generated predictions
├── Reg1/ # Linear regression with regularization
│ ├── g04_reg1.py # Main script: Ridge, Lasso, ElasticNet
│ ├── g04_reg1_test.py # Test script
│ ├── X_train_regression1.npy # Training features
│ ├── y_train_regression1.npy # Training targets
│ ├── X_test_regression1.npy # Test features
│ └── y_test_regression1.npy # Generated predictions
├── Reg2/ # Robust regression with RANSAC
│ ├── g04_reg2.py # Main script: RANSAC two-cluster regression
│ ├── g04_reg2_test.py # Test script
│ ├── X_train_regression2.npy # Training features
│ ├── y_train_regression2.npy # Training targets
│ ├── X_test_regression2.npy # Test features
│ └── y_test_regression2.npy # Generated predictions
├── Project_ML_2023_2024.pdf # Project specification document
├── .gitignore
└── README.md
```

## Requirements

- numpy
- matplotlib
- scikit-learn
- tensorflow / keras
- imbalanced-learn (imblearn)

### Data Requirements

The following data files must be present in their respective directories:

**Class1/**:

- `Xtrain_Classification1.npy` - Training images (28x28x3 RGB)
- `ytrain_Classification1.npy` - Training labels (binary)
- `Xtest_Classification1.npy` - Test images

**Class2/**:

- `Xtrain_Classification2.npy` - Training images (28x28x3 RGB)
- `ytrain_Classification2.npy` - Training labels (6 classes)
- `Xtest_Classification2.npy` - Test images

**Reg1/**:

- `X_train_regression1.npy` - Training features
- `y_train_regression1.npy` - Training targets
- `X_test_regression1.npy` - Test features

**Reg2/**:

- `X_train_regression2.npy` - Training features
- `y_train_regression2.npy` - Training targets
- `X_test_regression2.npy` - Test features

## Installation

1. Install dependencies as specified above:

```bash
pip install numpy matplotlib scikit-learn tensorflow imbalanced-learn
```

2. Prepare data files as required
3. Configure any necessary settings

## Usage

### Configuration

Each script has configurable flags at the top of the `main()` function:

**Class1 (`g04_class1.py`):**

- `path`: Data directory (default: 'Class1/')
- `grid_search`: Enable hyperparameter tuning (True/False)
- `bal_mode`: Class balancing method - 'smote', 'oversample', 'undersample', 'class_weight', 'aug_plus_os', 'double_minority', 'None'
- `chosen_model`: Model to run - 'log_reg', 'nb', 'svm', 'cnn'

**Class2 (`g04_class2.py`):**

- `path`: Data directory (default: 'Class2/')
- `bal_mode`: Class balancing method - 'smote', 'oversample', 'undersample', 'class_weight', 'None'
- `chosen_model`: Currently only 'cnn' implemented

**Reg1 (`g04_reg1.py`):**

- `path`: Data directory (default: 'Reg1/')
- `k_save`: CV folds for saving predictions (default: 5)
- `model_save`: Model to save predictions for - 'R' (Ridge), 'L' (Lasso), 'EN' (ElasticNet)
- `single_model`: Run only the specified model (True/False)

**Reg2 (`g04_reg2.py`):**

- `path`: Data directory (default: 'Reg2/')
- `scale`: Use StandardScaler (True/False)
- `test_alpha`: Test RANSAC thresholds (True/False)

### Running the Project

Run each problem independently:

```bash
# Classification Problem 1
cd Class1
python g04_class1.py

# Classification Problem 2
cd Class2
python g04_class2.py

# Regression Problem 1
cd Reg1
python g04_reg1.py

# Regression Problem 2
cd Reg2
python g04_reg2.py
```

### Optional Flags/Parameters

- **Class1/Class2**: Modify `bal_mode` to experiment with different class balancing strategies. Set `grid_search=True` for hyperparameter optimization (Class1 only).
- **Reg1**: Set `single_model=True` to run only one regression model. Change `model_save` to select which model's test predictions to save.
- **Reg2**: Set `test_alpha=True` to perform grid search over RANSAC residual thresholds.

## Outputs

Each script saves test predictions as `.npy` files in their respective directories:

- **Class1**: `ytest_Classification1.npy` - Binary predictions for test set
- **Class2**: `ytest_Classification2.npy` - 6-class predictions for test set
- **Reg1**: `y_test_regression1.npy` - Continuous predictions for test set
- **Reg2**: `y_test_regression2.npy` - Two-column array with predictions from both RANSAC clusters

Additional outputs during training:

- Training/validation loss and balanced accuracy plots (displayed via matplotlib)
- Model summaries (CNN architectures)
- Cross-validation SSE scores for Ridge/Lasso/ElasticNet
- Confusion matrices (Class2)

## Notes

- All scripts use a fixed random seed for reproducibility (seed=21 in Class2, none in Class1/Reg1/Reg2)
- Data is normalized to [0, 1] range for image classification tasks
- Reg1 uses StandardScaler for feature and target scaling
- Scripts print detailed progress metrics including class distributions, balanced accuracy scores, CV SSE, and training history
- Reg2 saves a two-column prediction array where each column corresponds to one RANSAC cluster's predictions
