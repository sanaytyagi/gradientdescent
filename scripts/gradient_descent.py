import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Salary_Data.csv')

currentM = 7000
currentB = 0

stepMax = 16000
learning_rate = 0.0001

#linear representation of data
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
    for i in range(stepMax):
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
x_vals = []
for i in range(12):
    x_vals.append(i)
y_vals = [f(currentM, x, currentB) for x in x_vals]
plt.plot(x_vals, y_vals, label="f(x) = 3x + 0")
plt.show()

#gradient descent
newM, newB = gradient_descent()

#plot loss
min_m = -100000
max_m = 100000
curr_m = min_m
m_vals = []
numMPoints = 50
while curr_m <= max_m:
    m_vals.append(curr_m)
    curr_m += (max_m - min_m) / numMPoints

plt.plot(m_vals, [loss(m, newB) for m in m_vals], label="m loss function")
plt.title("Loss Function for m")
plt.show()

min_b = -100000
max_b = 100000
curr_b = min_b
b_vals = []
numBPoints = 50
while curr_b <= max_b:
    b_vals.append(curr_b)
    curr_b += (max_b - min_b) / numBPoints
plt.plot(b_vals, [loss(newM, b) for b in b_vals])
plt.title("Loss Function for b")
plt.show()

print("Optimal M Value: " + str(newM))
print("Optimal B Value: " + str(newB))

df.plot(x='YearsExperience', y='Salary')
y_vals = [f(newM, x, newB) for x in x_vals]
plt.plot(x_vals, y_vals)
plt.title("New fitted line")
plt.show()

print("Sum of Squared Residual: " + str(loss(newM, newB)))
