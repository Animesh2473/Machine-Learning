import numpy as np
import pandas as pd

# Equation should be  y = 2x + 3
def gradient_descent(x,y,lr=0.01,epoches = 10000):
    m = 0.0
    b = 0.0
    for epoch in range(epoches):
        y_pred = m*x + b
        error = y - y_pred
        cost = np.mean(error*2)

        dm = -2*np.mean(error*x)
        db = -2*np.mean(error)

        b = b - db*lr
        m = m - dm*lr

        print(f"m = {m},b = {b},Epoches {epoch}: Cost = {cost}")


if __name__ == "__main__":
    x = np.array([1,2,3,4,5])
    y = np.array([5,7,9,11,13])
    gradient_descent(x,y)
