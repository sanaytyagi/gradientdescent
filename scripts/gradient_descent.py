import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/Salary_Data.csv')

# currentM = 8000
# currentB = 0
currentM = 13254.938454048064
currentB = 33496.33331671738

mStepMax = 1000
bStepMax = 10000
learning_rate = 0.0001

#linear approximation of data
def f(m, x, b):
    return m*x + b

#loss will be calculated by sum of squared residuals
def mLoss(m):
    rowCount = 0
    sumOfSquaredResiduals = 0
    for row in df.itertuples():
        yearsExperience = df.iloc[rowCount, 0]
        salary = df.iloc[rowCount, 1]
        predicted = f(m, yearsExperience, currentB)
        sumOfSquaredResiduals += (salary - predicted)**2
        rowCount += 1
    return sumOfSquaredResiduals

def bLoss(b):
    rowCount = 0
    sumOfSquaredResiduals = 0
    for row in df.itertuples():
        yearsExperience = df.iloc[rowCount, 0]
        salary = df.iloc[rowCount, 1]
        predicted = f(currentM, yearsExperience, b)
        sumOfSquaredResiduals += (salary - predicted)**2
        rowCount += 1
    return sumOfSquaredResiduals

def gradient_descent_m():
    dldm=0
    m = currentM
    dldm = finddldm(m)
    for i in range(0, mStepMax):
        if(abs(dldm) <= 0.001):
            print("found correct m")
            return m
        else:
            m = m - learning_rate * dldm
            dldm = finddldm(m)
    print("Could not find a valid m before step max was reached.")

def gradient_descent_b():
    dldb = 0
    b = currentB
    dldb = finddldb(b)
    for i in range(0, bStepMax):
        if(abs(dldb) <= 0.001):
            print("found correct b")
            return b
        else:
            b = b - learning_rate * dldb
            dldb = finddldb(b)
    print("Could not find a valid b before step max was reached")

def finddldm(m):
    rowCount = 0
    dldm = 0
    for row in df.itertuples():
        x=df.iloc[rowCount, 0]
        y=df.iloc[rowCount,1]
        dldm += x*(y-(m*x+currentB))
        rowCount += 1
    dldm *= -2
    return dldm

def finddldb(b):
    rowCount = 0
    dldb = 0
    for row in df.itertuples():
        x=df.iloc[rowCount, 0]
        y=df.iloc[rowCount,1]
        dldb += y-(currentM*x+b)
        rowCount += 1
    dldb *= -2
    return dldb
            
# plotting salary data vs linear approximation
df.plot(x='YearsExperience', y='Salary')
x_vals = np.linspace(0, 11)
y_vals = f(currentM, x_vals, currentB)
plt.plot(x_vals, y_vals, label="f(x) = 3x + 0")
plt.show()

#plot loss function for m
# min_m = -100000
# max_m = 100000
# m_vals = np.linspace(min_m, max_m)
# m_loss_vals = []

#for m in m_vals:
#    lossValue = mLoss(m)
#    m_loss_vals.append(lossValue)

#print(loss_vals)

#plt.plot(m_vals, m_loss_vals, label="m loss function")
#plt.show()

#plot loss function for b
# min_b = -100000
# max_b = 100000
# b_vals = np.linspace(min_b, max_b)
# b_loss_vals = []

# for b in b_vals:
#     lossValue = bLoss(b)
#     b_loss_vals.append(lossValue)
# print(b_vals)

# plt.plot(b_vals, b_loss_vals, label="b loss function")
# plt.show()



#gradient descent implementation
newM = gradient_descent_m()
newB = gradient_descent_b()
print(newM)
print(newB)
print(mLoss(newM))
print(bLoss(newB))