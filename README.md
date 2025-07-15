# Machine Learning Algorithm Demos

This repository contains Python scripts demonstrating core machine learning algorithms using simple datasets. Each script is self-contained and focuses on a specific algorithm, including model training, prediction, evaluation, and visualization.

## Algorithms Included

- **Linear Regression (`llinear.py` or `linear.py`)**
  - Demonstrates linear regression on a sample dataset.
  - Shows model fitting, visualization, and evaluation.

- **Logistic Regression (`logistic.py` or `logisitc.py`)**
  - Demonstrates logistic regression for classification.
  - Includes model training, prediction, and performance metrics.

- **Decision Tree (ID3 Algorithm) (`decisiontree.py`)**
  - Implements the ID3 decision tree algorithm.
  - Uses a dataset to build the decision tree and classifies new samples.

- **Naive Bayes Algorithm (`Naive Bayes algorithm.py`)**
  - Implements the Naive Bayes classifier.
  - Uses the Iris dataset.
  - Prints both correct and incorrect predictions.

- **k-Nearest Neighbor Algorithm (`k-Nearest Neighbor algorithm.py`)**
  - Implements the k-Nearest Neighbor (k-NN) classifier.
  - Uses the Iris dataset.
  - Prints both correct and incorrect predictions.

- **k-Means Algorithm (`k-Means algorithm.py`)**
  - Implements the k-Means clustering algorithm.
  - Groups data points into clusters and visualizes the results.

- **Hierarchical Clustering (`Hierarchical clustering .py`)**
  - Implements hierarchical clustering.
  - Demonstrates dendrogram creation and cluster assignment.

- **Support Vector Machine (`Support Vector.py`)**
  - Demonstrates SVM for classification.
  - Shows prediction accuracy and example outputs.

- **Random Forest (`Random Forest.py`)**
  - Implements random forest classifier.
  - Shows ensemble predictions and feature importance.

- **Local Outlier Factor (LOF) (`Lof.py`)**
  - Implements LOF for outlier detection.
  - Detects anomalies in dataset and visualizes results.

- **ARIMA Time Series Forecasting (`ARIMA.py`)**
  - Demonstrates time series forecasting using the ARIMA model.
  - Plots forecasts vs actuals.

- **Moving Average (`moving average.py`)**
  - Demonstrates time series smoothing and prediction using the moving average method.
  - Plots the original and smoothed series.

---

## Getting Started

### Prerequisites

You will need Python 3.x and the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `statsmodels`

To install the dependencies, run:
```sh
pip install numpy pandas matplotlib scikit-learn seaborn statsmodels
```

### Running the Programs

Clone the repository:
```sh
git clone https://github.com/Sangu111/ML.git
cd ML
```

Run any script using Python:
```sh
python llinear.py
python logistic.py
python decisiontree.py
python "Naive Bayes algorithm.py"
python "k-Nearest Neighbor algorithm.py"
python "k-Means algorithm.py"
python "Hierarchical clustering .py"
python "Support Vector.py"
python "Random Forest.py"
python Lof.py
python ARIMA.py
python "moving average.py"
```
*(Use the actual file name present in your directory; alternative names may exist, e.g., `linear.py` for linear regression, `logisitc.py` for logistic regression, etc.)*

---

## File Descriptions

- **llinear.py / linear.py:** Demonstrates linear regression, including data loading, training, prediction, and visualization.
- **logistic.py / logisitc.py:** Demonstrates logistic regression for classification, including model training and evaluation.
- **decisiontree.py:** Implements the ID3 algorithm for decision trees. Builds the tree from data and classifies new samples.
- **Naive Bayes algorithm.py:** Uses the Naive Bayes algorithm on the Iris dataset and displays both correct and incorrect predictions.
- **k-Nearest Neighbor algorithm.py:** Uses the k-NN algorithm on the Iris dataset, showing both correct and incorrect predictions.
- **k-Means algorithm.py:** Implements the k-Means clustering algorithm, groups data points, and visualizes the clusters.
- **Hierarchical clustering .py:** Demonstrates hierarchical clustering, including dendrogram construction and forming clusters.
- **Support Vector.py:** Demonstrates SVM for classification, displaying support vectors and output.
- **Random Forest.py:** Implements random forest classifier, showing feature importance and voting.
- **Lof.py:** Implements LOF for outlier detection and visualization.
- **ARIMA.py:** Shows ARIMA time series forecasting and plots.
- **moving average.py:** Demonstrates moving average for time series smoothing and prediction.

---

## Contributing

Contributions, improvements, and additional algorithms are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is open source and available under the MIT License.

---

**Tip:**  
Only the first 10 files in your repo are shown via the GitHub API at a time. To see your full file list, use the GitHub web UI:  
[View all files in GitHub UI](https://github.com/Sangu111/ML/search?q=py)

Let me know if you want any more customization or help!
