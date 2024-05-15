import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# função geradora
X = np.linspace(0,1,100)
modelo_gerador = np.sin(2*np.pi*X)

# amostras
x15 = np.linspace(0,1,15)
t15 = np.sin(2*np.pi*x15) + np.random.normal(0, 0.25, size=15)

x100 = np.linspace(0,1,100)
t100 = np.sin(2*np.pi*x100) + np.random.normal(0, 0.25, size=100)

# regressão
poly = PolynomialFeatures(degree=9) 
A = poly.fit_transform(x15.reshape(-1, 1)) # calculo da matriz A
model = LinearRegression() # criação do modelo
model.fit(A, t15) # treinamento
A = poly.fit_transform(X.reshape(-1, 1))
y15=model.predict(A) # predição

poly = PolynomialFeatures(degree=9) 
A = poly.fit_transform(x100.reshape(-1, 1)) # calculo da matriz A
model = LinearRegression() # criação do modelo
model.fit(A, t100) # treinamento
A = poly.fit_transform(X.reshape(-1, 1))
y100=model.predict(A) # predição

# resultados
fig, axs = plt.subplots(1, 2)

axs[0].plot(X,modelo_gerador,color='green')
axs[0].scatter(x15, t15, facecolors='none', edgecolors="blue")
axs[0].plot(X,y15,color='red')
axs[0].set_title('N=15')
axs[0].set_xlabel('x')
axs[0].set_ylabel('t')


axs[1].plot(X,modelo_gerador,color='green')
axs[1].scatter(x100, t100, facecolors='none', edgecolors="blue")
axs[1].plot(X,y100,color='red')
axs[1].set_title('N=100')
axs[1].set_xlabel('x')
axs[1].set_ylabel('t')

fig.legend(['Modelo Gerador','Dados de treinamento', 'Regressão'])
plt.show()
