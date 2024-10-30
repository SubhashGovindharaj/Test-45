import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset

data = pd.read_csv('BigMart Sales Data.csv')

# Handling missing values
# Impute missing values for 'Item_Weight' with mean and 'Outlet_Size' with the most frequent value
imputer_mean = SimpleImputer(strategy='mean')
imputer_freq = SimpleImputer(strategy='most_frequent')

data['Item_Weight'] = imputer_mean.fit_transform(data[['Item_Weight']])
data['Outlet_Size'] = imputer_freq.fit_transform(data[['Outlet_Size']]).ravel()


# Standardizing 'Item_Fat_Content' labels to have consistent values
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

# Encoding categorical variables
label_encoder = LabelEncoder()

# Apply label encoding to 'Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content'
label_cols = ['Item_Identifier', 'Outlet_Identifier', 'Item_Fat_Content']
for col in label_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Apply one-hot encoding to other categorical features
categorical_cols = ['Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
data = pd.get_dummies(data, columns=categorical_cols)

# Split the data into features and target variable
X = data.drop(columns=['Item_Outlet_Sales'])
y = data['Item_Outlet_Sales']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Train Random Forest Regressor model
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred_linear = linear_reg.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

# Evaluate model performance
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print evaluation results
print(f"Linear Regression - MSE: {mse_linear}, R2: {r2_linear}")
print(f"Random Forest Regressor - MSE: {mse_rf}, R2: {r2_rf}")
