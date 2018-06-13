# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import interp
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter("ignore")

np.random.seed(10)

# =============================================================================
# Funciones de Soporte (Generacion de Graficas)
# =============================================================================

def plot_confusion_matrix(Y_Test, Predictions, Score):
	cm = metrics.confusion_matrix(Y_Test, Predictions)
	plt.figure(figsize = (9, 9))
	sns.heatmap(cm, annot = True, fmt = ".3f", linewidths = .5, square = True, cmap = 'Blues_r');
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label');
	all_sample_title = 'Accuracy Score: {0}'.format(Score)
	plt.title(all_sample_title, size = 15);
	plt.show()

def plot_ROC_multiclass(X_Test, Y_Test, model):
	Y_Test = preprocessing.label_binarize(Y_Test, np.unique(Y_Test))
	n_classes = Y_Test.shape[1]
	Y_Scores = model.predict_proba(X_Test)
	lw = 2
	fpr = dict()
	tpr = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = metrics.roc_curve(Y_Test[:,i], Y_Scores[:,i])
	fpr["Micro"], tpr["Micro"], _ = metrics.roc_curve(Y_Test.ravel(), Y_Scores.ravel())
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	mean_tpr /= n_classes
	fpr["Macro"] = all_fpr
	tpr["Macro"] = mean_tpr
	plt.figure()
	plt.plot(fpr["Micro"], tpr["Micro"], label = 'Lower Average', color = 'deeppink', linestyle = ':', linewidth = 4)
	plt.plot(fpr["Macro"], tpr["Macro"], label = 'Upper Average', color = 'navy', linestyle = ':', linewidth = 4)
	colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'crimson', 'lightpink', 'darkgreen', 'salmon', 'sienna', 'bisque'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color = color, lw = lw, label = 'ROC class {}'.format(i))
	plt.plot([0, 1], [0, 1], 'k--', lw = lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positives')
	plt.ylabel('True Positives')
	plt.title('ROC Multiclass')
	plt.legend(loc = "lower right")

def print_results(adjustModel):
	print("Mejor valor de la validación cruzada: {:.4f}".format(adjustModel.best_score_))
	print("Mejores parámetros: {}".format(adjustModel.best_params_))

# =============================================================================
# Funcion Principal (Main)
# =============================================================================

def main():
	xtrain = np.loadtxt('datos/X_train.txt')
	ytrain = np.loadtxt('datos/y_train.txt')
	xtest = np.loadtxt('datos/X_test.txt')
	ytest = np.loadtxt('datos/y_test.txt')
	tuned_parameters = [{'n_estimators':[80,100,150,200], 'criterion':['gini','entropy']}]
	rfc = RandomForestClassifier()
	grid = GridSearchCV(rfc, tuned_parameters, cv = 5, scoring = 'accuracy')
	grid.fit(xtrain,ytrain)
	print("Parámetros óptimos:\n")
	print(grid.best_params_,"\n")
	print("Tabla de resultados de los parámetros:\n")
	for params, mean_score, scores in grid.grid_scores_:
		print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
	rfc2 = RandomForestClassifier(**grid.best_params_)
	rfc2.fit(xtrain,ytrain)
	print("error con rfc en train = ", rfc2.score(xtrain,ytrain))
	print("error con rfc en test = ", rfc2.score(xtest,ytest))
	prediction = rfc2.predict(xtest)
	score = rfc2.score(xtest, ytest)
	plot_confusion_matrix(ytest, prediction, score)
	prediction2 = rfc2.predict(xtrain)
	score2 = rfc2.score(xtrain, ytrain)
	plot_confusion_matrix(ytrain, prediction2, score2)
	plot_ROC_multiclass(xtest, ytest, rfc2)
    
if __name__ == "__main__":
    main()
