import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris-Experiment")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start run
with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="IrisModel"
    )

    print("Accuracy:", acc)
    print("Model Registered Successfully!")