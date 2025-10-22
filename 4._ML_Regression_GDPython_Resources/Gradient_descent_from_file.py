import numpy as np
import pandas as pd

# Equation should be  y = 2x + 3
def gradient_descent(x,y,lr=0.01,epoches = 100000):
    b, m = 0.0, 0.0


    # Scale x and y using Min-Max Scaling
    x_min , x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x -x_min)/(x_max-x_min)
    y_scaled = (y-y_min)/(y_max-y_min)

    for epoch in range(epoches):
        y_pred = m*x_scaled + b
        error = y_scaled - y_pred
        cost = np.mean(error*2)

        dm = -2*np.mean(error * x_scaled)
        db = -2*np.mean(error)

        b -= db*lr
        m -= dm*lr



        print(f"m = {m},b = {b},Epoches {epoch}: Cost = {cost}")


if __name__ == "__main__":
    df = pd.read_csv("home_prices.csv")
    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()
    gradient_descent(x,y)


