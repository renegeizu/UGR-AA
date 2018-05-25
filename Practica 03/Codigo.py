# -*- coding: utf-8 -*-

import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import interp
from sklearn import metrics, preprocessing
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

random_seed = 20000913

# =============================================================================
# Funciones de Soporte (Carga de Datos y Generacion de Graficas)
# =============================================================================

def load_dataset_opt():
    opt_train_x = np.float64(np.load("datos/optdigits_tra_X.npy"))
    opt_train_y = np.float64(np.load("datos/optdigits_tra_y.npy"))
    opt_test_x = np.float64(np.load("datos/optdigits_tes_X.npy"))
    opt_test_y = np.float64(np.load("datos/optdigits_tes_y.npy"))
    return opt_train_x, opt_train_y, opt_test_x, opt_test_y

def load_dataset_airfoil():
    X = np.load('datos/airfoil_self_noise_X.npy')
    y = np.load('datos/airfoil_self_noise_y.npy')
    airfoil_train_x, airfoil_test_x, airfoil_train_y, airfoil_test_y = train_test_split(X, y, test_size = 0.20, random_state = random_seed)
    return airfoil_train_x, airfoil_train_y, airfoil_test_x, airfoil_test_y

def plot_confusion_matrix(Y_Test, Predictions, Score):
    cm = metrics.confusion_matrix(Y_Test, Predictions)
    plt.figure(figsize = (9 ,9))
    sns.heatmap(cm, annot = True, fmt = ".3f", linewidths = .5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(Score)
    plt.title(all_sample_title, size = 15);
    plt.show()

def plot_ROC_multiclass(X_Test, Y_Test, model):
    Y_Test = preprocessing.label_binarize(Y_Test, np.unique(Y_Test))
    n_classes = Y_Test.shape[1]
    Y_Scores = model.decision_function(X_Test)
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
    plt.show()

def print_results(adjustModel):
    print(" Best Cross-Validation Value: {:.4f}".format(adjustModel.best_score_))
    print(" Best Parameters: {}".format(adjustModel.best_params_))
    print("\n Grid scores on development set:\n")
    means = adjustModel.cv_results_['mean_test_score']
    stds = adjustModel.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, adjustModel.cv_results_['params']):
        print(" %0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# =============================================================================
# Funciones Clasificacion (Preprocesamiento de Datos y Ajuste del Modelo)
# =============================================================================

def preprocessing_data_opt(X, y, Iterations):
    MinMaxScaler().fit_transform(X)
    tuned_parameters = [{'penalty':['l1', 'l2'], 'C':[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    clf = GridSearchCV(LogisticRegression(random_state = random_seed, max_iter = Iterations), tuned_parameters, cv = 5, scoring = 'accuracy')
    return clf.fit(X, y)
    
def adjust_data_opt(Y_Test, X_Test, adjustModel):
    predictions = adjustModel.predict(X_Test)
    scores = adjustModel.score(X_Test, Y_Test)
    return predictions, scores

# =============================================================================
# Funciones Regresion (Preprocesamiento de Datos y Ajuste del Modelo)
# =============================================================================

def ridge_model(X_Train, Y_Train):
    tuned_parameters_Ridge = [{'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    scores ='r2'
    ridge = Ridge()
    clf = GridSearchCV(ridge, tuned_parameters_Ridge, cv = 5, scoring = scores)               
    clf.fit(X_Train, Y_Train)
    return clf

def lasso_model(X_Train, Y_Train):
    tuned_parameters_Lasso = [{'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], 'selection':['random', 'cyclic'], 'tol':[1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}]
    scores ='r2'
    lasso = Lasso()
    clf = GridSearchCV(lasso, tuned_parameters_Lasso, cv = 5, scoring = scores)               
    clf.fit(X_Train, Y_Train)
    return clf

def preprocessing_data_airfoil(X_Train, X_Test, Y_Train, model):
    pipe = Pipeline([('Polynomial', preprocessing.PolynomialFeatures(degree = 6)), ('Scale', preprocessing.MinMaxScaler())])
    pipe.fit(X_Train)
    X_Train = pipe.transform(X_Train)
    X_Test = pipe.transform(X_Test)
    return ridge_model(X_Train, Y_Train)  

def adjust_data_airfoil(X_Train, Y_Train, X_Test, Y_Test, model, adjustModel, titulo):
    print_results(adjustModel)
    modelo = model(**adjustModel.best_params_)
    modelo.fit(X_Train, Y_Train)
    print("\n Error con ", titulo, " en Train = ", modelo.score(X_Train, Y_Train))
    print(" Error con ", titulo, " en Test = ", modelo.score(X_Test, Y_Test))  

# =============================================================================
# Funciones Ejercicios
# =============================================================================

def classification_problem():
    Train_X, Train_Y, Test_X, Test_Y = load_dataset_opt()
    model = preprocessing_data_opt(Train_X, Train_Y, 1000)
    prediction, score = adjust_data_opt(Test_Y, Test_X, model)
    plot_confusion_matrix(Test_Y, prediction, score)
    print_results(model)
    plot_ROC_multiclass(Test_X, Test_Y, model)
    
def regression_problem(regression_model, model, title):
    Train_X, Train_Y, Test_X, Test_Y = load_dataset_airfoil()
    clf = preprocessing_data_airfoil(Train_X, Test_X, Train_Y, regression_model)
    adjust_data_airfoil(Train_X, Train_Y, Test_X, Test_Y, model, clf, title)

# =============================================================================
# Funcion Principal (Main)
# =============================================================================

def main():
    np.random.seed(random_seed)
    print("\n Problema de Clasificacion: \n")
    classification_problem()
    input("\nPulsa Enter para continuar la ejecucion:")
    print("\n Problema de Regresion con Ridge: \n")
    regression_problem(ridge_model, Ridge, "Ridge")
    input("\nPulsa Enter para continuar la ejecucion:")
    print("\n Problema de Regresion con Lasso: \n")
    regression_problem(lasso_model, Lasso, "Lasso")
    
if __name__ == "__main__":
    main()