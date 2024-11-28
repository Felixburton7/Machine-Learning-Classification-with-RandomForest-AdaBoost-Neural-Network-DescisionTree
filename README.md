# Machine-Learning-Classification Task using a series of techniques 

## Disclaimer

This project was developed and graded using Google Colab, utilizing the free access to GPUs provided by the platform. Leveraging Google Colab’s free GPUs helps accelerate tasks like training large models or performing computationally intensive operations.

For a cleaner, more structured version of the code, please refer to the Scikit-learn file, which contains a more refined and organized version of the project.

Thank you for understanding! Feel free to contribute or raise issues if you have any suggestions or improvements.

## Description

This project involves creating a Decision Tree Classifier, Adaboost, Random Forest, and Neural Network, for a given dataset. The scores were posted onto Kaggle, and the results for adaboost ranked in the top 5% in the cohort (>250 members) with 98.7% accuracy when measured against an unseen dataset.  The classifier is trained using sample weights based on specific features, and its performance is evaluated using accuracy metrics and confusion matrix visualization.

## Example of the Decision Tree Classfier that achieved high accuracy on unseen data

## Installation

Ensure you have the necessary libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

### Data Preparation

Handle missing values and prepare the data for training and validation:

``'python
from sklearn.impute import SimpleImputer
import numpy as np

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
train_data[features] = imputer.fit_transform(train_data[features])
validation_data[features] = imputer.transform(validation_data[features])
test_data[features] = imputer.transform(test_data[features])

# Define sample weights based on 'viewed' and 'explored' features
sample_weight = np.ones(len(train_data))
sample_weight += train_data['viewed'] * 10  # Adjust the multiplier as needed
sample_weight += train_data['explored'] * 10  # Adjust the multiplier as needed
```

### Model Training and Prediction

Create and train the Decision Tree classifier with sample weights:

```python
from sklearn.tree import DecisionTreeClassifier

# Create a decision tree with a max_depth of 4 and random_state = 7
dt = DecisionTreeClassifier(max_depth=4, random_state=7)

# Fit the training data to decision tree
dt.fit(train_data[features], train_data[target], sample_weight=sample_weight)

# Predict on the validation set
dt_y_pred = dt.predict(validation_data[features])
```

### Model Evaluation

Evaluate the model using accuracy and visualize the confusion matrix:

```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Evaluate the model
dt_accuracy = accuracy_score(validation_data[target], dt_y_pred)
print("Decision Tree Accuracy:", dt_accuracy)

# Visualize the confusion matrix
cm = confusion_matrix(validation_data[target], dt_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()
```

### Additional Notes

- The `max_depth` & `random_state` parameters of the Decision Tree Classifier can be adjusted as needed.
- The multipliers for the `viewed` and `explored` features can be fine-tuned based on the specific dataset and requirements.
- Ensure the dataset is split appropriately into training, validation, and test sets.
  

## Conclusion

This project demonstrates the process of building a Decision Tree Classifier with custom sample weights, evaluating its performance, and visualizing the results using a confusion matrix. The accuracy metric helps in understanding the model's prediction capability on the validation dataset.
