# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#Semilla Random
second_impact = 20000913 

# =============================================================================
# Codigo - Ejercicio sobre la búsqueda iterativa de óptimos
# Gradiente Descendente
# =============================================================================

#Funcion E(u,v)
def E_Function(points):
	u = points[0]
	v = points[1]
	return np.float64(((u**3*np.e**(v-2))-(4*v**3*np.e**(-u)))**2)

#Funcion E'(u,v)
def dE_Function(points):
	u = points[0]
	v = points[1]
	points = np.array([0.,0.],np.float64)
	points[0] = 2 * ((u**3*np.e**(v-2))-(4*v**3*np.e**(-u))) * ((3*np.e**(v-2)*u**2)+(4*v**3*np.e**(-u)))
	points[1] = 2 * ((u**3*np.e**(v-2))-(4*v**3*np.e**(-u))) * ((u**3*np.e**(v-2))-(12*np.e**(-u)*v**2))
	return points

#Funcion f(x,y)
def F_Function(points):
	x = points[0]
	y = points[1]
	return np.float64((x-2)**2+2*(y+2)**2+2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

#Funcion f'(x,y)
def dF_Function(points):
	x = points[0]
	y = points[1]
	points = np.array([0.,0.],np.float64)
	points[0] = (2*x)+(4*np.pi*np.sin(2*np.pi*y)*np.cos(2*np.pi*x))-4
	points[1] = (4*y)+(4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y))+8
	return points

#Pintar graficas de lineas
def graphic_plot(X, Y, X_Label, Y_Label, title):
	plt.plot(X,Y)
	plt.xlabel(X_Label)
	plt.ylabel(Y_Label)
	plt.title(title)
	plt.show()

#Funcion del gradiente descendente
def Gradient_Descent(function, derivate, w, learning_rate, iterations, verbose, graphic, minimum, error=-np.inf):
	X = np.array([])
	Y = np.array([])
	for it in range(iterations):
		w = w - learning_rate * derivate(w)
		result = np.float64(function(w))
		if verbose:
			print("Iteraciones:", it+1, "Error:", result)
		if graphic:
				X = np.append(X, it)
				Y = np.append(Y, result)
		if minimum:
			if it == 0:
				Wa = w
				Werror = result
				Wit = 1
			elif Werror > result:
				Wa = w
				Werror = result
				Wit = it+1
		if result < error:
			break
	if minimum:
		print("Iteraciones: ", Wit, ", Puntos: ", Wa, "Error:", Werror)
	else:
		print("Iteraciones: ", it+1, ", Puntos: ", w, "Error:", result)
	if graphic:
		graphic_plot(X, Y, "Iteraciones", "Error", "Grafica Gradiente Descendente")

# =============================================================================
# Codigo - Ejercicio sobre Regresión Lineal
# =============================================================================

def coef2line(w):
	if(len(w)!= 3):
		raise ValueError('Solo se aceptan rectas para el plano 2d. Formato: [<a0>, <a1>, <b>].')
	a = -w[0]/w[1]
	b = -w[2]/w[1]
	return a, b

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
	
def simula_unif(N=2, dims=2, size=(0, 1)):
	m = np.random.uniform(low=size[0], high=size[1], size=(N, dims))
	return m

#Funcion f(x1,x2)=sign((x1-0.2)^2+x2^2-0.6)
def Fn_function(x1,x2):
	return np.sign((x1-0.2)**2 + x2**2 - 0.6)

def label_data(x1, x2):
	y = Fn_function(x1,x2)
	idx = np.random.choice(range(y.shape[0]), size=(int(y.shape[0]*0.1)), replace=True)
	y[idx] *= -1
	return y

#Funcion de la PseudoIversa
def pseudoInverse(X,Y):
	return np.dot(np.linalg.pinv(X), Y)

#Funcion Calculo SGD
def function_SGD(X,Y,w):
	return 2.0*X*(np.dot(X,w)-Y)

#Funcion del Gradiente Descendente Estocastico
def SGD(X,Y, w, learning_rate, iterations):
	for n in range(iterations):
		for i in range(X.shape[0]):
			w = w - learning_rate*function_SGD(X[i],Y[i],w)
	return w

#Funcion que computa el error
def function_error(w, X, Y):
	error = 0
	for i in range(X.shape[0]):
		value = np.dot(np.transpose(w), X[i])
		if (value*Y[i]) < 0:
			error += 1.
	return error/X.shape[0]

# =============================================================================
# Codigo Principal
# =============================================================================

#Ejercicio 1.2
def ichi():
	print("Gradiente Descendente para E(u,v) con learning rate de 0.05")
	Gradient_Descent(E_Function, dE_Function, np.array([1.,1.],np.float64), 0.05, 100000000000, True, False, False, np.float64(10**(-14)))

#Ejercicio 1.3a - Parte 1
def ni():
	print("Gradiente Descendente para F(x,y) con learning rate de 0.01")
	Gradient_Descent(F_Function, dF_Function, np.array([1.,1.],np.float64), 0.01, 50, False, True, False)

#Ejercicio 1.3a - Parte 2
def san():
	print("Gradiente Descendente para F(x,y) con learning rate de 0.1")
	Gradient_Descent(F_Function, dF_Function, np.array([1.,1.],np.float64), 0.1, 50, False, True, False)

#Ejercicio 1.3b - Parte 1
def shi():
	print("Gradiente Descendente para F(x,y) con learning rate de 0.01 en distintos puntos")
	Gradient_Descent(F_Function, dF_Function, np.array([2.1,-2.1],np.float64), 0.01, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([3.,-3.],np.float64), 0.01, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([1.5,1.5],np.float64), 0.01, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([1.,-1.],np.float64), 0.01, 50, False, False, True)

#Ejercicio 1.3b - Parte 2
def go():
	print("Gradiente Descendente para F(x,y) con learning rate de 0.1 en distintos puntos")
	Gradient_Descent(F_Function, dF_Function, np.array([2.1,-2.1],np.float64), 0.1, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([3.,-3.],np.float64), 0.1, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([1.5,1.5],np.float64), 0.1, 50, False, False, True)
	Gradient_Descent(F_Function, dF_Function, np.array([1.,-1.],np.float64), 0.1, 50, False, False, True)

#Ejercicio 2.1
def roku():
	print("PseudoInversa y Gradiente Descendente Estocastico")
	np.random.seed(second_impact)
	#Cargamos los datos
	X_Train = np.load('datos/X_train.npy')
	Y_Train = np.load('datos/Y_train.npy')
	X_Test = np.load('datos/X_test.npy')
	Y_Test = np.load('datos/Y_test.npy')
	#Preparamos los datos
	Train_X = np.concatenate((X_Train[Y_Train == 1], X_Train[Y_Train == 5]))
	Train_Y = np.concatenate((Y_Train[Y_Train == 1], Y_Train[Y_Train == 5]))
	Test_X = np.concatenate((X_Test[Y_Test == 1], X_Test[Y_Test == 5]))
	Test_Y = np.concatenate((Y_Test[Y_Test == 1], Y_Test[Y_Test == 5]))
	Train_Y[Train_Y == 5] = -1
	Test_Y[Test_Y == 5] = -1
	#Insertar columna de 1
	Train_X = np.c_[Train_X, np.ones((Train_X.shape[0],))]
	Test_X = np.c_[Test_X, np.ones((Test_X.shape[0],))]
	#Mezclamos los datos
	idx = np.arange(0, Train_X.shape[0], dtype=np.int32)
	np.random.shuffle(idx)
	Train_X = Train_X[idx]
	Train_Y = Train_Y[idx]
	#PseudoInversa
	w = pseudoInverse(Train_X, Train_Y)
	error = function_error(w, Train_X, Train_Y)
	print("Error PseudoInversa Train: ", error)
	error = function_error(w, Test_X, Test_Y)
	print("Error PseudoInversa Test: ", error)
	graphic_scatter(Train_X, Train_Y, w, 'Grafica PseudoInversa Train')
	graphic_scatter(Test_X, Test_Y, w, 'Grafica PseudoInversa Test')
	#SGD
	w = SGD(Train_X, Train_Y, np.ones((3,)), 0.01, 100)
	error = function_error(w, Train_X, Train_Y)
	print("Error SGD Train: ", error)
	error = function_error(w, Test_X, Test_Y)
	print("Error SGD Test: ", error)
	graphic_scatter(Train_X, Train_Y, w, 'Grafica SGD Train')
	graphic_scatter(Test_X, Test_Y, w, 'Grafica SGD Test')

#Ejercicio 2.2
def nana():
	print("Complejidad del Modelo Linea")
	np.random.seed(second_impact)
	error_in = 0
	error_out = 0
	for i in range(1000):
		#Generamos los datos
		X = simula_unif(N=1000, dims=2, size=(-1, 1))
		y = label_data(X[:, 0], X[:, 1])
		X = np.c_[X, np.ones((X.shape[0],))]
		#Calculamos el Gradiente
		w = SGD(X, y, np.ones((3,)), 0.01, 100)
		error = function_error(w, X, y)
		print("Error IN de la iteracion ", i+1, ": ", error)
		error_in += error
		#Calculamos el error de los test
		X = simula_unif(N=1000, dims=2, size=(-1, 1))
		y = label_data(X[:, 0], X[:, 1])
		X = np.c_[X, np.ones((X.shape[0],))]
		error = function_error(w, X, y)
		print("Error OUT de la iteracion ", i+1, ": ", error)
		error_out += error
	graphic_scatter(X, y, w, 'Grafica SGD')
	print("Error IN Medio de las 1000 iteraciones: ", error_in/1000)
	print("Error OUT Medio de las 1000 iteraciones: ", error_out/1000)
		

def main():
	#Ejercicio 1.2
	ichi()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 1.3a - Parte 1
	ni()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 1.3a - Parte 2
	san()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 1.3b - Parte 1
	shi()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 1.3b - Parte 2
	go()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 2.1
	roku()
	input("Pulsa Enter para continuar la ejecucion:")
	#Ejercicio 2.2
	nana()

if __name__ == "__main__":
	main()