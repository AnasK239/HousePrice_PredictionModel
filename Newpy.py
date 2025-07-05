import numpy as np
import matplotlib.pyplot as plt
import copy, math


def main():
    
    X_training, y_training = load_data()
    X_features = ['size(sqft)','bedrooms','floors','age']

    X_normalized ,X_mu ,X_sigma = z_score_normalization(X_training)
    w_norm, b_norm, hist = gradient_descent(X_normalized, y_training, np.zeros(X_training.shape[1]), 0.0, 1.0e-1, 10000)

    m = X_normalized.shape[0]
    Predictions = np.zeros(m)
    print("---------------------------------------------------------------------------")
    choice = input("For visual plots type (plt) ,, To predict a house price type (prd):  ").strip().lower()
    if choice != 'plt':
        UserInput = input("Enter a house's features as comma-separated values (size(sqft), bedrooms, floors, age): ")
        try:
            user_features = np.array([float(x) for x in UserInput.split(',')])
            user_features_normalized = (user_features - X_mu) / X_sigma
            user_prediction = Predict_Price(user_features_normalized, w_norm, b_norm)
            print("---------------------------------------------------------------------------")
            print(f"Predicted price for the house: ${user_prediction * 1000:.2f}")
        except ValueError:
            print("Invalid input. Please enter numeric values separated by commas.")
    else:
        for i in range(m):
            Predictions[i] = np.dot(X_normalized[i], w_norm) + b_norm

        fig , ax = plt.subplots(1,4, figsize=(12,3) , sharey=True)
        for i in range(len(ax)):
            ax[i].scatter(X_training[:,i],y_training , label = 'Actual Price')
            ax[i].set_xlabel(X_features[i])
            ax[i].scatter(X_training[:,i], Predictions, color='red', label='Predictions')

        ax[0].set_ylabel('Price ($1000s)')
        ax[0].legend()
        plt.show()



def Predict_Price(X, w, b):
    """
    Predict the price of a house given its features.
    X: numpy array of shape (n_features,)
    w: numpy array of weights
    b: bias term
    """
    return np.dot(X, w) + b

def load_data():

    data = np.loadtxt("houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

def compute_gradients(X, y, w, b):

    m,n = X.shape
    dj_dw = np.zeros((n,)) 
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i],w)+b - y[i])
        for j in range (n):
            dj_dw[j] += err * X[i,j]
        dj_db += err

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def compute_cost(X,y, w ,b):

    cost =0.0
    m = X.shape[0]

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2
    cost /= (2*m)

    return np.squeeze(cost)

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):

    w= copy.deepcopy(w_in)  
    b = b_in
    cost_history = []

    for i in range(num_iters):
        dj_dw , dj_db = compute_gradients(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:  
            cost_history.append(compute_cost(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:6d} : Cost: {cost_history[-1]:8.2f} | w0: {w[0]:6.2f} |  w1: {w[1]:6.2f} |  w2: {w[2]:6.2f} |  w3: {w[3]:6.2f} |  b: {b:6.2f}")

    print(f"w,b found by gradient descent: w: {w}, b: {b:0.2f}")
    return w, b, cost_history

def z_score_normalization(X):

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X-mu) / sigma

    return X_norm, mu, sigma


if __name__ == "__main__":
    main()