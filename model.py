# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cluster import KMeans

# Read the CSV file as a data frame
df = pd.read_csv('/home/project_docker/res_dpre.csv')

# Implement the naive bayes algorithm

price_threshold = df['Price_ln'].median()
df['Price_category'] = (df['Price_ln'] > price_threshold).astype(int)

feature_columns =['Suburb','Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',"CouncilArea","Type","Method","SellerG","Postcode"
                ,"Lattitude","Longtitude","Regionname","Propertycount","Year"]
X = df[feature_columns]
y = df['Price_category']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


naive_bayes_model = GaussianNB()

naive_bayes_model.fit(X_train, y_train)

y_pred = naive_bayes_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")
# Calculate R²
r_squared = r2_score(y_test, y_pred)
print(f"R²: {r_squared:.4f}")
# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.4f}")
# Calculate RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

naive_result = f"naive bayes :\nScore: {accuracy}\nMean Absolute Error: {mae}\n r2 : {r_squared }\n mean_squared_error {mse}\nRMSE {rmse}\n\n"

# Implement the decision tree algorithm

price_threshold = df['Price_ln'].median()
df['Price_category'] = (df['Price_ln'] > price_threshold).astype(int)

feature_columns = ['Rooms', 'Distance', 'Bathroom', 'Landsize', 'BuildingArea',
                    "Type",  "Lattitude", "Longtitude", 
                    "Propertycount"]
X = df[feature_columns]
y = df['Price_category']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42)

param_grid_dt_extended = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

grid_search_dt_extended = GridSearchCV(estimator=decision_tree_model, param_grid=param_grid_dt_extended, scoring='accuracy', cv=5)

grid_search_dt_extended.fit(X_train, y_train)

best_params_dt_extended = grid_search_dt_extended.best_params_
print('Best Parameters for Extended Decision Tree:', best_params_dt_extended)

best_dt_model_extended = grid_search_dt_extended.best_estimator_
y_pred_best_dt_extended = best_dt_model_extended.predict(X_test)

# Evaluate the best Extended Decision Tree model
accuracy_best_dt_extended = accuracy_score(y_test, y_pred_best_dt_extended) * 100
print(f"Accuracy with Best Extended Decision Tree Model: {accuracy_best_dt_extended:.2f}%")
r_squared = r2_score(y_test, y_pred_best_dt_extended)
print(f"R²: {r_squared:.4f}")
mae = mean_absolute_error(y_test, y_pred_best_dt_extended)
print(f"MAE: {mae:.4f}")
mse = mean_squared_error(y_test, y_pred_best_dt_extended)
print(f"MSE: {mse:.4f}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

tree_result = f" decision tree :\nScore: {accuracy_best_dt_extended}\nMean Absolute Error: {mae}\n r2 : {r_squared }\n mean_squared_error {mse}\nRMSE {rmse}\n\n"

# Implement the KNeighbors algorithm
knn_model = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  
}

grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn, scoring='accuracy', cv=5)
grid_search_knn.fit(X_train, y_train)

best_params_knn = grid_search_knn.best_params_
print('Best Parameters for KNN:', best_params_knn)

best_knn_model = grid_search_knn.best_estimator_
y_pred_best_knn = best_knn_model.predict(X_test)

# Evaluate the best KNN model
accuracy_best_knn = accuracy_score(y_test, y_pred_best_knn) * 100
print(f"Accuracy with Best KNN Model: {accuracy_best_knn:.2f}%")
# Calculate R²
r_squared = r2_score(y_test, y_pred_best_knn)
print(f"R²: {r_squared:.4f}")

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred_best_knn)
print(f"MAE: {mae:.4f}")

# Calculate MSE
mse = mean_squared_error(y_test, y_pred_best_knn)
print(f"MSE: {mse:.4f}")

# Calculate RMSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")

knn_result = f" knn :\nScore: {accuracy_best_knn}\nMean Absolute Error: {mae}\n r2 : {r_squared }\n mean_squared_error {mse}\nRMSE {rmse}\n\n"

# Implement the neural_network algorithm


price_threshold = df['Price_ln'].median()

df['Price_category'] = (df['Price_ln'] > price_threshold).astype(int)

feature_columns = ['Rooms', 'Distance', 'Bathroom', 'Landsize', 'BuildingArea',
                    "Type",  "Lattitude", "Longtitude", 
                    "Propertycount"]
X = df[feature_columns]
y = df['Price_category']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100,),  
    activation='relu',           
    solver='adam',               
    max_iter=1000,               
    random_state=42
)


mlp_model.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Evaluate the MLPClassifier model
accuracy_mlp = accuracy_score(y_test, y_pred_mlp) * 100
print(f"Accuracy with MLPClassifier (Neural Network): {accuracy_mlp:.2f}%")
mae = mean_absolute_error(y_test, y_pred_mlp)
mse = mean_squared_error(y_test, y_pred_mlp)
r2 = r2_score(y_test, y_pred_mlp)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

neural_network_result = f" neural_network :\nScore: {accuracy_mlp}\nMean Absolute Error: {mae}\n r2 : {r_squared }\n mean_squared_error {mse}\n\n"

# Implement the Random forest algorithm

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
rf_model.score(X_train, y_train)
random_forest_result = f" random forest:\nScore: {rf_model.score}\n r2 : {r_squared }\n mean_squared_error {mse}\n\n"

# Implement the linear regression and k-means algorithm


feature_columns = ['Suburb', 'Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt', "CouncilArea", "Type", "Method", "SellerG", "Postcode",
                    "Lattitude", "Longtitude", "Regionname", "Propertycount", "Year"]
X = df[feature_columns]
y = df['Price_ln']
X = pd.get_dummies(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  
X['Cluster'] = kmeans.fit_predict(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso_model = Lasso(alpha=0.01)  
lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(X_test)
threshold = 0.5  
y_pred_binary = (y_pred > threshold).astype(int)
# Evaluate the model
rmse = sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)
y_test_binary = (y_test > threshold).astype(int)
y_pred_train = lasso_model.predict(X_train)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Absolute Error (MAE):', mae)
print('Mean Squared Error (MSE):', mse)
print('R-squared:', r2)

# Calculate accuracy
accuracy = accuracy_score(y_test_binary, y_pred_binary) * 100
print(f"Accuracy: {accuracy:.2f}%")
threshold = 0.5  # Adjust the threshold as needed
y_pred_train_binary = (y_pred_train > threshold).astype(int)

# Evaluate the model on the training set
rmse_train = sqrt(mean_squared_error(y_train, y_pred_train))
print('Root Mean Squared Error (Training):', rmse_train)
y_pred_train = lasso_model.predict(X_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print('MAE (Training):', mae_train)
print('MSE (Training):', mse_train)
print('R-squared (Training):', r2_train)
# Convert true labels to binary
y_train_binary = (y_train > threshold).astype(int)

# Calculate accuracy for the training set
accuracy_train = accuracy_score(y_train_binary, y_pred_train_binary) * 100
print(f"Accuracy (Training): {accuracy_train:.2f}%")

linear_regression_result = f"linear regression:\nScore: {accuracy}\nMean Absolute Error: {mae}\n r2 : {r2 }\n mean_squared_error {mse}\nRMSE {rmse}\n\n"
linear_regression_train_result = f" linear regression :\nScore: {accuracy_train}\nMean Absolute Error: {mae_train}\n r2 : {r2_train}\n mean_squared_error {mse_train}\nRMSE {rmse_train}\n\n"

all_result = naive_result + tree_result + knn_result + neural_network_result + random_forest_result + linear_regression_result + linear_regression_train_result
with open("/home/project_docker/Models_Result.txt", 'w') as file:
    file.write(all_result)

print("Results have been saved to Models_Result.txt")
