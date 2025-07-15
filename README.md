Here is your fully updated README.md file, now including the two recent algorithms—k-Means and Hierarchical Clustering—making a total of seven main algorithms. File names have been clarified based on your repo contents.

---

# ML Algorithms

This repository contains Python implementations of classic machine learning algorithms, each demonstrated with suitable datasets. The goal is to provide clear and practical examples for learning and reference.

## Contents

1. **Linear Regression** (`linear.py` or `llinear.py`)
    - Demonstrates linear regression on a sample dataset.
    - Shows model fitting, visualization, and evaluation.

2. **Logistic Regression** (`logisitc.py` or `logistic.py`)
    - Demonstrates logistic regression for binary classification.
    - Includes model training, prediction, and performance metrics.

3. **Decision Tree (ID3 Algorithm)** (`decision tree.py` or `decisiontree.py`)
    - Implements the ID3 decision tree algorithm.
    - Uses a dataset to build the decision tree and classifies a new sample.

4. **Naive Bayes Algorithm** (`Naive Bayes algorithm.py`)
    - Implements the Naive Bayes classifier.
    - Uses the Iris dataset.
    - Prints both correct and incorrect predictions.

5. **k-Nearest Neighbor Algorithm** (`k-Nearest Neighbor algorithm.py`)
    - Implements the k-Nearest Neighbor (k-NN) classifier.
    - Uses the Iris dataset.
    - Prints both correct and incorrect predictions.

6. **k-Means Algorithm** (`k-Means algorithm.py`)
    - Implements the k-Means clustering algorithm.
    - Groups data points into clusters and visualizes the results.

7. **Hierarchical Clustering** (`Hierarchical clustering .py`)
    - Implements hierarchical clustering.
    - Demonstrates dendrogram creation and cluster assignment.

---

## Getting Started

### Prerequisites

You will need Python 3.x and the following packages:
- numpy
- pandas
- matplotlib
- scikit-learn
- seaborn
- scipy

To install the dependencies, run:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy
```

### Running the Programs

Clone the repository:
```bash
git clone https://github.com/Sangu111/ML.git
cd ML
```

Run any script using Python:
```bash
python llinear.py
python logistic.py
python decisiontree.py
python "Naive Bayes algorithm.py"
python "k-Nearest Neighbor algorithm.py"
python "k-Means algorithm.py"
python "Hierarchical clustering .py"
```

### Dataset Organization

All datasets are organized in the `dataset/` folder:
- `iris.csv` - Iris flower dataset
- `iris_naivebayes.csv` - Iris dataset for Naive Bayes
- `PlayTennis.csv` - Tennis playing decision dataset
- `clean_study_hours_vs_marks.csv` - Study hours vs marks dataset
- `Mall_Customers.csv` - Customer segmentation dataset
- `ecommerce_customers.csv` - E-commerce customer dataset

---

## File Descriptions

- **llinear.py**: Demonstrates linear regression, including data loading, training, prediction, and visualization using the `clean_study_hours_vs_marks.csv` dataset.
- **logistic.py**: Demonstrates logistic regression for binary classification, including model training and evaluation using the `iris.csv` dataset.
- **decisiontree.py**: Implements the ID3 algorithm for decision trees. Builds the tree from the `PlayTennis.csv` dataset and classifies new samples.
- **Naive Bayes algorithm.py**: Uses the Naive Bayes algorithm on the `iris_naivebayes.csv` dataset and displays both correct and incorrect predictions.
- **k-Nearest Neighbor algorithm.py**: Uses the k-NN algorithm on the `iris_naivebayes.csv` dataset, showing both correct and incorrect predictions.
- **k-Means algorithm.py**: Implements the k-Means clustering algorithm using the `Mall_Customers.csv` dataset, groups data points, and visualizes the clusters.
- **Hierarchical clustering .py**: Demonstrates hierarchical clustering using the `ecommerce_customers.csv` dataset, including dendrogram construction and forming clusters.

---

## Contributing

Contributions, improvements, and additional algorithms are welcome! Feel free to open issues or submit pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Tip:**  
Only the first 10 files in your repo are shown via the GitHub API at a time. To see your full file list, use the GitHub web UI: [View all files in GitHub UI](https://github.com/Sangu111/ML/tree/main)

Let me know if you want any more customization or help!
