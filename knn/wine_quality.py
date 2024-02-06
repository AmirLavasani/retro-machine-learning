import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def read_data(file_name):
    # Load the dataset with semicolon delimiter
    data = pd.read_csv(file_name, delimiter=';')
    return data


def draw_quality_histogram(data):
    # Plot distribution histogram of the 'quality' label
    plt.hist(data['quality'], bins=6, alpha=0.7, color='blue')
    plt.xlabel('Quality')
    plt.ylabel('Frequency')
    plt.title('Distribution of Wine Quality')
    plt.show()


def split_data(data):
    # Extract features and labels
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=30
    )
    return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, k = 5):
    # Implement KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the train set
    train_preds = knn.predict(X_train)

    # Calculate and return accuracy on train set
    train_accuracy = accuracy_score(y_train, train_preds)
    return train_accuracy, knn


def test_model(model, X_test, y_test):
    # Predict on the test set
    test_preds = model.predict(X_test)

    
    # plot_fitted_model(X_test, test_preds)

    # Calculate and return accuracy on test set
    test_accuracy = accuracy_score(y_test, test_preds)
    return test_accuracy


def plot_fitted_model(X_test, test_preds):
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values  # Convert DataFrame to array

    cmap = sns.cubehelix_palette(as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap)
    f.colorbar(points)
    plt.show()


def find_best_k_with_grid_search(X_train, y_train, X_test, y_test, param_grid):
    # Create a KNN classifier
    knn = KNeighborsClassifier()

    # Perform GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameter
    best_k = grid_search.best_params_['n_neighbors']

    # Fit model with best k on entire training set
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)

    # Calculate accuracy on train and test set
    train_accuracy = best_knn.score(X_train, y_train)
    test_accuracy = best_knn.score(X_test, y_test)
    
    return grid_search.best_params_, train_accuracy, test_accuracy


def calculate_bagged_knn(X_train, y_train, X_test, y_test, best_k, best_weights):
    # Create a KNeighborsClassifier with best parameters
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights)
    
    # Create BaggingClassifier with KNeighborsClassifier as base estimator
    bagged_knn = BaggingClassifier(estimator=knn, n_estimators=100, max_samples=0.3)
    
    # Fit the BaggingClassifier on the training data
    bagged_knn.fit(X_train, y_train)
    
    # Calculate training and test accuracies
    train_accuracy = bagged_knn.score(X_train, y_train)
    test_accuracy = bagged_knn.score(X_test, y_test)

    return train_accuracy, test_accuracy


# Example usage:

# Read data
data = read_data('./knn/wine_quality/winequality-red.csv')

# Draw quality histogram
# draw_quality_histogram(data)

# Split data
X_train, X_test, y_train, y_test = split_data(data)


# with static k value
k_value = 15
print(f"1. with static k value = {k_value}")

# Fit model and get accuracy on train data
train_accuracy, knn_model = fit_model(X_train, y_train, k_value)
print(f"Train Accuracy k={k_value}: {train_accuracy*100:.2f}%")

# Get accuracy on test data
test_accuracy = test_model(knn_model, X_test, y_test)
print(f"Test Accuracy k={k_value}: {test_accuracy*100:.2f}%")


# with gird search to find the best k value
print(f"\n {'*'*20}")
print(f"2. with grid search for k value")

# Assuming you have X_train, y_train variables
max_k=50

# Define a grid of hyperparameters
parameters = {'n_neighbors': range(2, max_k + 1)}

best_params, train_accuracy, test_accuracy = find_best_k_with_grid_search(X_train, y_train, X_test, y_test, parameters)
print(f"Best k: {best_params['n_neighbors']}")
print(f"Train Accuracy with Best k: {train_accuracy*100:.2f}%")
print(f"Test Accuracy with Best k: {test_accuracy*100:.2f}%")


# with weighted gird search to find the best k value
print(f"\n {'*'*20}")
print(f"3. with grid search for weighted knn")

# Assuming you have X_train, y_train variables
max_k=50

# Define a grid of hyperparameters
parameters = {
    "n_neighbors": range(2, max_k + 1),
    "weights": ["uniform", "distance"]
}
best_params, train_accuracy, test_accuracy = find_best_k_with_grid_search(X_train, y_train, X_test, y_test, parameters)
print(f"Best k: {best_params['n_neighbors']}")
print(f"Train Accuracy with Best k: {train_accuracy*100:.2f}%")
print(f"Test Accuracy with Best k: {test_accuracy*100:.2f}%")


# Improving on kNN in scikit-learn With Bagging
print(f"\n {'*'*20}")
print(f"4. Ensemble with bagging")
train_accuracy, test_accuracy = calculate_bagged_knn(X_train, y_train, X_test, y_test, best_params['n_neighbors'], best_params['weights'])
print(f"Train Accuracy with Bagged KNN: {train_accuracy*100:.2f}%")
print(f"Test Accuracy with Bagged KNN: {test_accuracy*100:.2f}%")