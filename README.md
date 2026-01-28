I would like to use the classic Iris dataset, which contains measurements of 150 iris flowers from three species (setosa, versicolor, and virginica). Each observation has four numeric features: sepal length, sepal width, petal length, and petal width. My goal is to use these features to classify the species of iris using a support vector machine (SVM). I plan to split the data into training and test sets, standardize the features, and then fit SVM models with different kernels (such as linear and radial). I will compare their performance using test error, confusion matrices, and possibly cross-validation to choose good tuning parameters. Finally, I will interpret which features seem most important and discuss how well SVM separates the three classes.

# Iris Classification (PCA + SVM)

A small machine learning project using the classic **Iris dataset** to:
- visualize feature importance with **PCA (biplot)**  
- train and tune **multi-class SVM** models (linear / RBF)
- evaluate performance with confusion matrices + accuracy

## What’s inside
- `Iris_Classification_Project.ipynb` — the full notebook (EDA → PCA → SVM tuning → evaluation)

## Methods
- **Dataset:** `sklearn.datasets.load_iris`
- **Split:** 50% train / 50% test (`random_state=0`)
- **PCA:** standardize features → reduce to 2 components → biplot to see which features drive separation
- **SVM:** GridSearchCV (10-fold CV) to tune hyperparameters:
  - Linear: `C`
  - RBF: `C`, `gamma`
- **Evaluation:** accuracy + confusion matrix (all models perfectly separate *setosa*; most errors occur between *versicolor* and *virginica*)

## Results (this run)
- Best tuned SVM achieved ~**97% test accuracy** (≈2 errors out of 75).

## Package
- numpy
- matplotlib
- scikit-learn



