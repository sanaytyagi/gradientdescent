import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Salary_Data.csv')

currentM = 7000
currentB = 0

stepMax = 20000
learning_rate = 0.0001

#linear approximation of data
def f(m, x, b):
    return m*x + b

#loss calculated by sum of squared residuals
def loss(m, b):
    rowCount = 0
    sumOfSquaredResiduals = 0
    for row in df.itertuples():
        yearsExperience = df.iloc[rowCount, 0]
        salary = df.iloc[rowCount, 1]
        predicted = f(m, yearsExperience, b)
        sumOfSquaredResiduals += (salary - predicted)**2
        rowCount += 1
    return sumOfSquaredResiduals

def gradient_descent():
    b = currentB
    m = currentM
    dldm = findderivatives(m, b)[0]
    dldb = findderivatives(m, b)[1]
    for i in range(0, stepMax):
        if(abs(dldm) <= 0.001 and abs(dldb) <= 0.001):
            print("found correct m and b")
            return m, b
        else:
            m = m - learning_rate * dldm
            b = b - learning_rate * dldb
            dldm = findderivatives(m, b)[0]
            dldb = findderivatives(m,b)[1]
    print("Step max reached")

def findderivatives(m, b):
    rowCount = 0
    dldb = 0
    dldm = 0
    for row in df.itertuples():
        x=df.iloc[rowCount, 0]
        y=df.iloc[rowCount,1]
        dldm += x*(y-(m*x+b))
        dldb += y-(m*x+b)
        rowCount += 1
    dldm *= -2
    dldb *= -2
    return dldm, dldb

            
# plotting salary data vs linear approximation
df.plot(x='YearsExperience', y='Salary')
x_vals = np.linspace(0, 11)
y_vals = f(currentM, x_vals, currentB)
plt.plot(x_vals, y_vals, label="f(x) = 3x + 0")
plt.show()

#gradient descent
newM, newB = gradient_descent()

#plot loss
min_m = -100000
max_m = 100000
m_vals = np.linspace(min_m,max_m)
plt.plot(m_vals, [loss(m, newB) for m in m_vals], label="m loss function")
plt.title("Loss Function for m")
plt.show()

min_b = -100000
max_b = 100000
b_vals= np.linspace(min_b,max_b)
plt.plot(b_vals, [loss(newM, b) for b in b_vals])
plt.title("Loss Function for b")
plt.show()

print("Optimal M Value: " + str(newM))
print("Optimal B Value: " + str(newB))

actualOptimalValues = np.polyfit(df['YearsExperience'], df['Salary'], 1)
print("Actual Optimal M Value: " + str(actualOptimalValues[0]))
print("Actual Optimal B Value: " + str(actualOptimalValues[1]))

print("Sum of Squared Residual: " + str(loss(newM, newB)))
