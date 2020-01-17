import numpy as np
import matplotlib.pyplot as plt
from tools import *

def gauss(num, move):
    A = 1/num
    step = move/(num+1)
    def gauss_sum(x):
        y = 0
        for i in range(1,num+1):
            y += A*np.power(np.e, -np.power(x-step*i,2)*0.5/np.power(5000,2))
        return y
    return gauss_sum

#sep
move = au2km(as2au(3.45,3.09816061924855))
aperture = au2km(as2au(7,3.09816061924855))
afr = 0.84
print(aperture)

#nov
move = au2km(as2au(6.49,2.41562442021788))
aperture = au2km(as2au(6,2.41562442021788))
afr = 1.05
print(aperture)

#dec
move = au2km(as2au(5.67,2.04435148882201))
aperture = au2km(as2au(7,2.04435148882201))
afr = 0.95
print(move)

g_move = gauss(50, move)
g_still = gauss(1, move)
x = np.linspace(move/2-aperture, move/2+aperture,1001)

step = x[1] - x[0]
sum_move = 0
sum_still = 0
for i in x:
    sum_move += step*g_move(i)
    sum_still += step*g_still(i)
print('if move: '+str(sum_move))
print('if not move: '+str(sum_still))
print((sum_still-sum_move)/sum_still*100)

y1 = [g_move(i) for i in x]
y2 = [g_still(i) for i in x]
plt.plot(x, y1)
plt.plot(x, y2)
plt.show()