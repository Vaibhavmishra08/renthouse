import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('houses.csv')

# Example: Convert 'location' (categorical) into numerical format
# (Assume location is a categorical variable, and we need to one-hot encode it)
data = pd.get_dummies(data, columns=['location'], drop_first=True)

# Features (X) and target (y)
X = data[['size', 'bedrooms', 'location_1', 'location_2']]  # example locations after encoding
y = data['rent']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the rent prices using the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r_squared = model.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r_squared}')

# Make a prediction for a new house
new_house = pd.DataFrame({'size': [1200], 'bedrooms': [3], 'location_1': [1], 'location_2': [0]})
predicted_rent = model.predict(new_house)
print(f'Predicted rent for new house: ${predicted_rent[0]:.2f}')
