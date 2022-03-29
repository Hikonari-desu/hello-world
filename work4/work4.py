import numpy as np
import time
import matplotlib.pyplot as plt


Loss_list=[]
x=np.array([1,3])#输入
y=np.array([0.95,0.05])#输出
w1=np.array([[1,2],[-2,0]])#权重一
w2=np.array([[1,1],[0,-2]])#权重二
a=1#学习率
Gamma=np.array([-3,1])#阈值一
Beta=np.array([2,3])#阈值二
Err=0.0001#均方误差门限
round=0#次数
#参数梯度赋予初值
grad_w1=np.array([[0,0],[0,0]])
grad_Gamma=np.array([0,0])
grad_Beta=np.array([0,0])
grad_w2=np.array([[0,0],[0,0]])

def sigmoid(x):#激活函数
    z=np.exp(-1*x)
    sig=1/(1+z)
    return sig

def err(y0):#误差函数
    global y
    err=np.dot(y-y0,y-y0)/len(y0)#求均方误差
    return err

#第一次处理
a1=np.dot(x,w1)
z1=sigmoid(a1-Gamma)
a2=np.dot(z1,w2)
z2=sigmoid(a2-Beta)
y0=z2
err0=err(y0)
Loss_list.append(err0)

#迭代
while err0>=Err and round<10000:#误差阈值门限判断
    '''
    if round%100==0:
        print(round)
        print(err0)
    '''
    round=round+1#计数+1
    #反向梯度传播
    g2=(z2-y)*z2*(1-z2)
    grad_Beta=-g2
    grad_w2=np.outer(z1,g2)
    e1=z1*(1-z1)*np.dot(g2,w2.T)
    grad_Gamma=-e1
    grad_w1=np.outer(x,e1)
    #梯度下降
    Beta=Beta-a*grad_Beta
    w2=w2-a*grad_w2
    Gamma=Gamma-a*grad_Gamma
    w1=w1-a*grad_w1
    #再次计算
    a1=np.dot(x,w1)
    z1=sigmoid(a1-Gamma)
    a2=np.dot(z1,w2)
    z2=sigmoid(a2-Beta)
    y0=z2
    #学习率修正
    temp_err=err(y0)
    if temp_err<err0:
        a=1.1*a
    elif temp_err>err0:
        a=0.65*a
    err0=temp_err
    Loss_list.append(err0)

print("W1:",w1)
print("W2:",w2)
print("Gamma:",Gamma)
print("Beta:",Beta)
print("Loss:",err0)
print("Round:",round)
    

plt.figure()
plt.xlabel("Round")
plt.ylabel("LOSS")
plt.plot(np.arange(len(Loss_list)),Loss_list,color='r',label="a")
#plt.plot(np.arange(len(Loss_list[1])),Loss_list[1],color='g',label="b")
#plt.plot(np.arange(len(Loss_list[2])),Loss_list[2],color='b',label="a=1.5")
plt.legend()
plt.show()