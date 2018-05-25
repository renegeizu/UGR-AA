# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# =============================================================================
# Codigo - Funciones Simula
# =============================================================================

#Simulacion de Datos - Distribucion Uniforme
def simula_unif(N = 2, dim = 2, rango = (0, 1)):
	return np.random.uniform(low = rango[0], high = rango[1], size = (N, dim))

#Simulacion de Datos - Distribucion Gauss
def simula_gaus(size, sigma, media = None):
	media = 0 if media is None else media
	if len(size) >= 2:
		N = size[0]
		size_sub = size[1:]
		out = np.zeros(size, np.float64)
		for i in range(N):
			out[i] = np.random.normal(loc = media, scale = np.sqrt(sigma), size = size_sub)
	else:
		out = np.random.normal(loc = media, scale = sigma, size = size)
	return out

#Simulacion de a y b - Recta f = ax + b
def simula_recta(intervalo = (-1, 1), ptos = None):
	if ptos is None:
		m = np.random.uniform(intervalo[0], intervalo[1], size = (2, 2))
	a = (m[0,1]-m[1,1]) / (m[0,0]-m[1,0])
	b = m[0,1] - a * m[0,0]
	return a, b

# =============================================================================
# Codigo - Funciones Graficas
# =============================================================================

def line2coef(a, b):
	w = np.zeros(3, np.float64)
	w[0] = -a
	w[1] = 1.0
	w[2] = -b
	return w

def coef2line(w):
	if(len(w)!= 3):
		raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
	a = -w[0]/w[1]
	b = -w[2]/w[1]
	return a, b

#Pintar Nube de Etiquetas y Pesos
def graphic_scatter(X, y, w, tittle):
	a, b = coef2line(w)
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = grid.dot(w)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$w^tx$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, cmap="RdYlBu", edgecolor='white', label='Datos')
	ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth=2.0, label='Solucion')
	ax.set(xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]), xlabel='Intensidad promedio', ylabel='Simetria')
	ax.legend()
	plt.title(tittle)
	plt.show()

#Pintar Nube de Etiquetas y Recta
def plot_datos_recta(X, y, a, b, title = 'Point clod plot', xaxis = 'x axis', yaxis = 'y axis'):
	w = line2coef(a, b)
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = grid.dot(w)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$w^tx$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c = y, s = 50, linewidth = 2, cmap = "RdYlBu", edgecolor = 'white', label = 'Datos')
	ax.plot(grid[:, 0], a*grid[:, 0]+b, 'black', linewidth = 2.0, label = 'Solucion')
	ax.set(xlim = (min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), ylim = (min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]), xlabel = xaxis, ylabel = yaxis)
	ax.legend()
	plt.title(title)
	plt.show()
	
#Pintar Nube de Puntos
def plot_cloud_points(X):
	plt.plot(X[:, 0], X[:, 1], 'bo')
	plt.show()

#Pintar Nube de Etiquetas y Funcion
def plot_datos_cuad(X, y, fz, fx1, title='Point clod plot', xaxis='x axis', yaxis='y axis'):
	min_xy = X.min(axis=0)
	max_xy = X.max(axis=0)
	border_xy = (max_xy-min_xy)*0.01
	xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
	grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
	pred_y = fz(grid)
	pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
	f, ax = plt.subplots(figsize=(8, 6))
	contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu', vmin=-1, vmax=1)
	ax_c = f.colorbar(contour)
	ax_c.set_label('$f(x, y)$')
	ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
	ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, cmap="RdYlBu", edgecolor='white', label='Datos')
	ax.set(xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]), xlabel=xaxis, ylabel=yaxis)
	ax.legend()
	plt.title(title)
	plt.show()
	
# =============================================================================
# Codigo Funciones
# =============================================================================

def A_Function(X):
	return (X[:, 0]-10)**2 + (X[:, 1]-20)**2 - 400

def A1_Function(X0):
	return 20-np.sqrt(-X0**2+20*X0+300)

def B_Function(X):
	return 0.5*(X[:, 0]+10)**2 + (X[:, 1]-20)**2 - 400

def B1_Function(X0):
	return 0.5*(40-np.sqrt(2)*np.sqrt(-X0**2-20*X0+700))

def C_Function(X):
	return 0.5*(X[:, 0]-10)**2 - (X[:, 1]+20)**2 - 400

def C1_Function(X0):
	return -(np.sqrt(X0**2-20*X0-700)/np.sqrt(2)) - 20

def D_Function(X):
	return X[:, 1] - 20*X[:, 0]**2 - 5*X[:, 0] + 3

def D1_Function(X0):
	return 20*X0**2 + 5*X0 - 3

# =============================================================================
# Codigo Perceptron - PLA
# =============================================================================

#Algoritmo Perceptron
def ajusta_PLA(datos, label, max_iter, w_ini):
	epsilon = 10**-6
	for t in range(max_iter):
		w = np.copy(w_ini)
		for i, x in enumerate(datos):
			if (np.dot(datos[i], w_ini)*label[i]) <= 0:
				w_ini = w_ini + datos[i]*label[i]
		if(np.sum(np.abs(w-w_ini)) < epsilon):
			print("Converge en la iteracion ", t+1)
			return w_ini
			break
	return w_ini

# =============================================================================
# Codigo SGD - Gradiente Descendente Estocastico
# =============================================================================

#Funcion del SGD
def function_SGD(X,y,w):
	a = np.dot(-y, X)	
	b = sigmoid(np.dot(np.dot(-y, np.transpose(w)), X))
	return (a * b)/X.shape[0]

#Error del SGD
def error_SGD(w, X, Y):
	error = 0.
	Y0 = np.copy(Y)
	Y0[Y == -1] = 0
	score = np.zeros(Y.shape)
	for i in range(X.shape[0]):
		value = sigmoid(np.dot(np.dot(Y0[i], np.transpose(w)), X[i]))
		score[i] = value
		if (value <= 0.5 and Y0[i] == 1) or (value > 0.5 and Y0[i] == 0):
			error += 1.
	return error/X.shape[0], score

# =============================================================================
# Codigo Regresion Logistica - RLSGD
# =============================================================================

#Sigmoide
def sigmoid(w):
	return (expit(w))

#Error Logistico log-likelihood
def likelihood(X, y, w):
    aux = np.dot(X, w)
    return np.sum(y * aux - np.log(1 + np.exp(aux)))

#Regresion Logistica
def logisticRegression(X, y, iterations = 10000, lr = 0.01):
	w = np.zeros(3)
	w_ini = np.zeros(3)
	idx = np.arange(X.shape[0])
	for i in range(iterations):
		np.random.shuffle(idx)
		X = X[idx]
		y = y[idx]
		batchX = X[0:50:]
		batchY = y[0:50:]
		for j in range(batchX.shape[0]):
			w = w - lr * function_SGD(batchX[j], batchY[j], w)
		if(np.linalg.norm(w_ini - w) < 0.01):
			return w
		w_ini = np.copy(w)
	return w

# =============================================================================
# Codigo PseudoInversa
# =============================================================================

def pseudoInverse(X,Y):
	return np.dot(np.linalg.pinv(X), Y)

def function_error(w, X, Y):
	error = 0
	for i in range(X.shape[0]):
		value = np.dot(np.transpose(w), X[i])
		if (value*Y[i]) < 0:
			error += 1.
	return error/X.shape[0]

# =============================================================================
# Codigo - PLA-Pocket
# =============================================================================

def PLA_Pocket(X, y, iterations):
	it = 30
	w = ajusta_PLA(X, y, it, np.zeros(3))
	w_ini = np.copy(w)
	for i in range(iterations):
		w = ajusta_PLA(X, y, it, w)
		error_w = function_error(w, X, y)
		error_w_ini = function_error(w_ini, X, y)
		if error_w < error_w_ini:
			w_ini = np.copy(w)
	return w_ini

# =============================================================================
# Codigo - Ejercicio 01
# =============================================================================

def ejecicio11():
	X = simula_unif(50, 2, (-50, 50))
	plot_cloud_points(X)
	X = simula_gaus((50, 2), [5, 7])
	plot_cloud_points(X)
	
def ejercicio12():
	contMe = 0
	contMa = 0
	a, b = simula_recta()
	print("\na: ", a, ", b: ", b)
	X = simula_unif(50, 2, (-50, 50))
	y1 = np.sign(X[:, 1]-a*X[:, 0]-b)
	plot_datos_recta(X, y1, a, b)
	y2 = np.copy(y1)
	while True:
		valor = int(np.random.randint(y2.shape[0]))
		if(y2[valor] == 1 and contMa != 2):
			y2[valor] *= -1
			contMa += 1
		elif(y2[valor] == -1 and contMe != 2):
			y2[valor] *= -1
			contMe += 1
		if(contMe == 2 and contMa == 2):
			break
	plot_datos_recta(X, y2, a, b)
	return X, y1, y2

def ejercicio13(X, y):
	plot_datos_cuad(X, y, A_Function, A1_Function)
	plot_datos_cuad(X, y, B_Function, B1_Function)
	plot_datos_cuad(X, y, C_Function, C1_Function)
	plot_datos_cuad(X, y, D_Function, D1_Function)
	
# =============================================================================
# Codigo - Ejercicio 02
# =============================================================================

def ejercicio21(X, y, y1, Iterations):
	print("\nEjercicio 2.1a:\n")
	w = np.zeros(3)
	print("Vector Pesos: ", w)
	ajusta_PLA(X, y, Iterations, w)
	for i in range(10):
		w = np.random.uniform(low=0, high=1, size=3)
		print("\nVector Pesos: ", w)
		ajusta_PLA(X, y, Iterations, w)
	print("\nEjercicio 2.1b:\n")
	w = np.zeros(3)
	print("Vector Pesos: ", w)
	ajusta_PLA(X, y1, Iterations, w)
	for i in range(10):
		w = np.random.uniform(low=0, high=1, size=3)
		print("\nVector Pesos: ", w)
		ajusta_PLA(X, y1, Iterations, w)
	
def ejercicio22():
	#Simulamos los parametros (train)
	a, b = simula_recta()
	X = simula_unif(100, 2, (-50, 50))
	X = np.c_[X, np.ones((X.shape[0],))]
	y = np.sign(X[:, 1]-a*X[:, 0]-b)
	#Se lanza la Regresion Logistica
	w = logisticRegression(X, y)
	#Se pintan los datos
	plot_datos_recta(X, y, a, b)
	#Se obtiene el error
	error, y_score = error_SGD(w, X, y)
	print("\nError de Clasificacion: ", error)
	print("Error Logistico: ", likelihood(X, y, w))
	print("Valores de w: ", w)
	#Simulamos los parametros (test) (misma funcion de etiquetado)
	X0 = simula_unif(2000, 2, (-50, 50))
	X0 = np.c_[X0, np.ones((X0.shape[0],))]
	y0 = np.sign(X0[:, 1]-a*X0[:, 0]-b)
	#Se pintan los datos
	plot_datos_recta(X0, y0, a, b)
	#Se obtiene el error
	error, y_score = error_SGD(w, X0, y0)
	print("Error de Clasificacion: ", error)
	print("Error Logistico: ", likelihood(X0, y0, w))
	#Simulamos los parametros (test) (distinta funcion de etiquetado)
	a, b = simula_recta()
	y0 = np.sign(X0[:, 1]-a*X0[:, 0]-b)
	#Se pintan los datos
	plot_datos_recta(X0, y0, a, b)
	#Se obtiene el error
	error, y_score = error_SGD(w, X0, y0)
	print("Error de Clasificacion: ", error)
	print("Error Logistico: ", likelihood(X0, y0, w))
	
def ejercicioBonus():
	#Carga de Datos
	X_Train = np.load('datos/X_train.npy')
	Y_Train = np.load('datos/Y_train.npy')
	X_Test = np.load('datos/X_test.npy')
	Y_Test = np.load('datos/Y_test.npy')
	#Preparacion
	Train_X = np.concatenate((X_Train[Y_Train == 4], X_Train[Y_Train == 8]))
	Train_Y = np.concatenate((Y_Train[Y_Train == 4], Y_Train[Y_Train == 8]))
	Test_X = np.concatenate((X_Test[Y_Test == 4], X_Test[Y_Test == 8]))
	Test_Y = np.concatenate((Y_Test[Y_Test == 4], Y_Test[Y_Test == 8]))
	Train_Y[Train_Y == 4] = -1
	Test_Y[Test_Y == 4] = -1
	Train_Y[Train_Y == 8] = 1
	Test_Y[Test_Y == 8] = 1
	#Columna de 1s
	Train_X = np.c_[Train_X, np.ones((Train_X.shape[0],))]
	Test_X = np.c_[Test_X, np.ones((Test_X.shape[0],))]
	#Mezcla de Datos
	idx = np.arange(0, Train_X.shape[0], dtype=np.int32)
	np.random.shuffle(idx)
	Train_X = Train_X[idx]
	Train_Y = Train_Y[idx]
	#PseudoInversa
	w = pseudoInverse(Train_X, Train_Y)
	error1 = function_error(w, Train_X, Train_Y)
	print("Error PseudoInversa Train: ", error1)
	error2 = function_error(w, Test_X, Test_Y)
	print("Error PseudoInversa Test: ", error2)
	graphic_scatter(Train_X, Train_Y, w, 'Grafica PseudoInversa Train')
	graphic_scatter(Test_X, Test_Y, w, 'Grafica PseudoInversa Test')
	#PLA-Pocket
	w = PLA_Pocket(Train_X, Train_Y, 20)
	error3 = function_error(w, Train_X, Train_Y)
	print("Error PLA Pocket Train: ", error3)
	error4 = function_error(w, Test_X, Test_Y)
	print("Error PLA Pocket Test: ", error4)
	graphic_scatter(Train_X, Train_Y, w, 'Grafica PLA Pocket Train')
	graphic_scatter(Test_X, Test_Y, w, 'Grafica PLA Pocket Test')
	#Cota de error
	e = 0.05
	print("Cota Error PseudoInversa Train: ", e/(error1-e))
	print("Cota Error PseudoInversa Test: ", e/(error2-e))
	print("Cota Error PLA Pocket Train: ", e/(error3-e))
	print("Cota Error PLA Pocket Test: ", e/(error4-e))
	
def main():
	np.random.seed(20000913)
	ejecicio11()
	input("\nPulsa Enter para continuar la ejecucion:")
	X, Y0, Y1 = ejercicio12()
	input("\nPulsa Enter para continuar la ejecucion:")
	ejercicio13(X, Y1)
	input("\nPulsa Enter para continuar la ejecucion:")
	ejercicio21(np.c_[X, np.ones((X.shape[0],))], Y0, Y1, 20)
	input("\nPulsa Enter para continuar la ejecucion:")
	ejercicio22()
	input("\nPulsa Enter para continuar la ejecucion:")
	ejercicioBonus()

if __name__ == "__main__":
	main()