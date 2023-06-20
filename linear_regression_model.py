import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.intercept = None #nilai defautlnya none dan diupdate ketika fitting
        self.coefficients = None #nilai defautlnya none dan diupdate ketika fitting
        self.loss_mse_history = [] #untuk menampung error MSE setiap epochnya
        self.loss_rmse_history = [] #untuk menampung error RMSE setiap epochnya
        self.loss_r_history = [] #untuk menampung error R^2 setiap epochnya

    def expand_features(self, X, degree):
        # matriks = X.shape # untuk mendapatkan ukuran fitur (1001,2)
        # print(matriks)
        n_samples, n_features = X.shape # n_samples = 1001, n_features = 2
        X_expanded = np.ones((n_samples, 1)) # menambahkan intercept pada expand fitur
        for i in range(n_features):
            for j in range(1, degree+1):
                X_expanded = np.hstack((X_expanded, np.power(X[:, i:i+1], j))) #menggabungkan intercept dengan perkalian derajat
        # if matriks[1] != 1:
        X_expanded = np.hstack((X_expanded,np.prod(X,axis=1).reshape(-1,1))) #menggabungkan itercept, perkalian derajat, dan perkalian kedua fitur
        return X_expanded

    def fit(self, X, y):
        n_samples, n_features = X.shape # untuk mendapatkan ukuran data (1001, 2)

        # Initialize intercept and coefficients
        self.intercept = 0 # bias awal adalah 0
        self.coefficients = np.zeros((n_features,)) #bobot awal 0 (2, 1) [0,0]

        # Gradient descent optimization
        for _ in range(self.epochs):
            y_pred = self.predict(X) # Memanggil method predict
            error = y_pred - y # Menghitung error

            # Update intercept
            self.intercept -= self.learning_rate * np.mean(error)

            # Update coefficients
            for i in range(n_features):
                self.coefficients[i] -= self.learning_rate * np.mean(error * X[:, i]) # X[:, i] data X (1001,)

            # Calculate evaluation metrics
            mse, rmse, r2 = self.evaluate(X, y)

            # Append values to lists
            self.loss_mse_history.append(mse) # menambahkan error setiap epochnya ke dalam variabel loss_mse_history
            self.loss_rmse_history.append(rmse) # menambahkan error setiap epochnya ke dalam variabel loss_rmse_history
            self.loss_r_history.append(r2) # menambahkan error setiap epochnya ke dalam variabel loss_r_history

    def predict(self, X):
        p = np.dot(X, self.coefficients) + self.intercept # kalkulasi untuk predict
        return p

    def evaluate(self, X, y):
        y_pred = self.predict(X) # memanggil method predict
        mse = np.mean((y - y_pred) ** 2) # klakulasi MSE
        rmse = np.sqrt(mse) # klakulasi RMSE
        r2 = self.calculate_r2(y, y_pred) # menanggil method r2 untuk kalkukasi r2
        return mse, rmse, r2

    def calculate_r2(self, y, y_pred):
        ss_total = np.sum((y - np.mean(y)) ** 2) # sigma kuadrat target asli - rata2 target asli
        ss_residual = np.sum((y - y_pred) ** 2) # sigma kuadrat target asli - rata2 prediksi
        if ss_total == 0:
            r2 = 1
        else:
            r2 = 1 - (ss_residual / ss_total)
        return r2