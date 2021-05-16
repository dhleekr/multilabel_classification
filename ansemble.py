import pandas as pd
import numpy as np

alphabets = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
a = pd.read_csv("./results/0.8649.csv")
b = pd.read_csv("./results/0.86189.csv")
c = pd.read_csv("./results/0.86329.csv")
d = pd.read_csv("./results/0.86329.csv")
e = pd.read_csv("./results/0.86552.csv")
f = pd.read_csv("./results/0.86569.csv")
g = pd.read_csv("./results/0.86569.csv")



submission = (a+b+c+d+e+f+g)/7
print(submission)
for i in range(5000):
    order = [] # 각 index에서의 확률값 전체
    sorted_order = []

    for alphabet in alphabets:
        order.append(submission[alphabet][i])
        if submission[alphabet][i] > 0.5:
            sorted_order.append(submission[alphabet][i])

    sorted_order = sorted(sorted_order, reverse=True) # 내림차순 정렬

    if len(sorted_order) > 14:
        sorted_order = sorted_order[:14]
    elif len(sorted_order) < 10:
        sorted_order = sorted(order, reverse=True)
        sorted_order = sorted_order[:10]

    idx = []
    for j in range(len(sorted_order)):
        idx.append(order.index(sorted_order[j]))
    
    for k in idx:
        submission[alphabets[k]][i] = 1

for alphabet in alphabets:
    submission[alphabet] = np.where(submission[alphabet]==1, 1, 0)

prediction = np.array(submission.iloc[:,1:])
sample = pd.read_csv("./sample_submission.csv")
sample.iloc[:,1:] = prediction
print(sample)
sample.to_csv("./results/ensemble.csv",index=False)