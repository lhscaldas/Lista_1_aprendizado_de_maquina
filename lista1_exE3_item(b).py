import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# função geradora
X = np.linspace(0,1,100)
modelo_gerador = np.sin(2*np.pi*X)

# amostra
x = np.linspace(0,0.5,10)
t = np.sin(2*np.pi*x) + np.random.normal(0, 0.25, size=10)

# regressão
poly0 = PolynomialFeatures(degree=0) 
A0 = poly0.fit_transform(x.reshape(-1, 1)) # calculo da matriz A
model0 = LinearRegression() # criação do modelo
model0.fit(A0, t) # treinamento
A0 = poly0.fit_transform(X.reshape(-1, 1))
y0=model0.predict(A0) # predição

poly1 = PolynomialFeatures(degree=1) 
A1 = poly1.fit_transform(x.reshape(-1, 1)) # calculo da matriz A
model1 = LinearRegression() # criação do modelo
model1.fit(A1, t) # treinamento
A1 = poly1.fit_transform(X.reshape(-1, 1))
y1=model1.predict(A1) # predição

poly3 = PolynomialFeatures(degree=3) 
A3 = poly3.fit_transform(x.reshape(-1, 1)) # calculo da matriz A
model3 = LinearRegression() # criação do modelo
model3.fit(A3, t) # treinamento
A3 = poly3.fit_transform(X.reshape(-1, 1))
y3=model3.predict(A3) # predição

poly9 = PolynomialFeatures(degree=9) 
A9 = poly9.fit_transform(x.reshape(-1, 1)) # calculo da matriz A
model9 = LinearRegression() # criação do modelo
model9.fit(A9, t) # treinamento
A9 = poly9.fit_transform(X.reshape(-1, 1))
y9=model9.predict(A9) # predição

# resultados
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(X,modelo_gerador,color='green')
axs[0, 0].scatter(x, t, facecolors='none', edgecolors="blue")
axs[0, 0].plot(X,y0,color='red')
axs[0, 0].set_title('M=0')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('t')
axs[0, 0].set_xlim(-0.1,1.1)
axs[0, 0].set_ylim(-2,2)

axs[0, 1].plot(X,modelo_gerador,color='green')
axs[0, 1].scatter(x, t, facecolors='none', edgecolors="blue")
axs[0, 1].plot(X,y1,color='red')
axs[0, 1].set_title('M=1')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('t')
axs[0, 1].set_xlim(-0.1,1.1)
axs[0, 1].set_ylim(-2,2)

axs[1, 0].plot(X,modelo_gerador,color='green')
axs[1, 0].scatter(x, t, facecolors='none', edgecolors="blue")
axs[1, 0].plot(X,y3,color='red')
axs[1, 0].set_title('M=3')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('t')
axs[1, 0].set_xlim(-0.1,1.1)
axs[1, 0].set_ylim(-2,2)

axs[1, 1].plot(X,modelo_gerador,color='green')
axs[1, 1].scatter(x, t, facecolors='none', edgecolors="blue")
axs[1, 1].plot(X,y9,color='red')
axs[1, 1].set_title('M=9')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('t')
axs[1, 1].set_xlim(-0.1,1.1)
axs[1, 1].set_ylim(-2,2)

fig.legend(['Modelo Gerador','Dados de treinamento', 'Regressão'])
plt.show()
