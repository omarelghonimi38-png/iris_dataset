from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


# تحميل بيانات Iris
iris = load_iris()

X = iris.data
y = iris.target


# إنشاء وتدريب الموديل
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


def predict(
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
):
    """
    تستقبل قياسات الزهرة وترجع اسم النوع المتوقع.
    """

    values = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction_number = model.predict(values)[0]

    species_name = iris.target_names[prediction_number]

    return species_name
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

 # Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

def predict(petal_width, petal_length, sepal_width, sepal_length):
  
    # Make a prediction based on the input features
    input_features = np.array([[petal_width, petal_length, sepal_width, sepal_length]])
    prediction = clf.predict(input_features)

    # Return the predicted species name
    return iris.target_names[prediction][0]