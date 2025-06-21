import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Let's use only a few features for now (keep it simple)
features = ['LotArea', 'BedroomAbvGr', 'YearBuilt', 'FullBath']
X = train[features]
y = train['SalePrice']

# Split the training data to test model performance
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict(X_val)
mse = mean_squared_error(y_val, val_preds)
print("Validation MSE:", mse)

# Predict on test data
X_test = test[features]
test_preds = model.predict(X_test)

# Load sample submission to get the ID format
submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = test_preds
submission.to_csv('my_submission.csv', index=False)
print("Submission file saved as my_submission.csv")
