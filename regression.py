from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        XT = X.T
        self.coefficient = np.linalg.inv(XT @ X) @ XT @ y
        if self.fit_intercept:
            self.intercept = self.coefficient[0]
            self.coefficient = self.coefficient[1:]
        else:
            self.intercept = 0

    def predict(self, X):
        return X @ self.coefficient + self.intercept

    def r2_score(self, y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def rmse(self, y, yhat):
        return np.sqrt(np.mean((y - yhat) ** 2))

f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]

X = np.array([f1, f2, f3]).T
y = np.array(y).reshape(-1, 1)

# CustomLinearRegression
regCustom = CustomLinearRegression(fit_intercept=True)
regCustom.fit(X, y)
y_pred_custom = regCustom.predict(X)
rmse_custom = regCustom.rmse(y, y_pred_custom)
r2_custom = regCustom.r2_score(y, y_pred_custom)

# sklearn LinearRegression
regSci = LinearRegression(fit_intercept=True)
regSci.fit(X, y)
y_pred_sci = regSci.predict(X)
rmse_sci = np.sqrt(mean_squared_error(y, y_pred_sci))
r2_sci = r2_score(y, y_pred_sci)

# Print the differences
print({
    "Intercept": float(regCustom.intercept) - regSci.intercept_,
    "Coefficient": regCustom.coefficient.ravel() - regSci.coef_,
    "R2": float(r2_custom) - float(r2_sci),
    "RMSE": float(rmse_custom) - float(rmse_sci),

})

