import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
data = pd.read_csv("hprice.csv")
print(data)
X = data.drop("price", axis=1)
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [4, 8, 16],
    "random_state": [42]
    }
rf_regressor = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, scoring="neg_mean_squared_error")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
print("RMSE de RandomForestRegressor:", rmse)
with open("model.pickle", "wb") as f:
    pickle.dump(model, f)
