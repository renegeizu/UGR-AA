# -*- coding: utf-8 -*-

import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import interp
from sklearn import metrics, preprocessing, svm
from sklearn.model_selection import GridSearchCV

random_seed = 20000913

# =============================================================================
# Funciones de Soporte (Carga de Datos y Generacion de Graficas)
# =============================================================================

def load_dataset():
    train_x = np.float64(np.loadtxt("datos/X_train.txt"))
    train_y = np.float64(np.loadtxt("datos/y_train.txt"))
    test_x = np.float64(np.loadtxt("datos/X_test.txt"))
    test_y = np.float64(np.loadtxt("datos/y_test.txt"))
    return train_x, train_y, test_x, test_y

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
# Funciones SVM (Preprocesamiento de Datos y Ajuste del Modelo)
# =============================================================================

def preprocessing_data(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    return grid_search.fit(X, y)
    
def adjust_data(Y_Test, X_Test, adjustModel):
    predictions = adjustModel.predict(X_Test)
    scores = adjustModel.score(X_Test, Y_Test)
    return predictions, scores

# =============================================================================
# Funciones Ejercicios
# =============================================================================

def svm_problem():
    Train_X, Train_Y, Test_X, Test_Y = load_dataset()
    model = preprocessing_data(Train_X, Train_Y, 5)
    prediction, score = adjust_data(Test_Y, Test_X, model)
    plot_confusion_matrix(Test_Y, prediction, score)
    print_results(model)
    plot_ROC_multiclass(Test_X, Test_Y, model)

# =============================================================================
# Funcion Principal (Main)
# =============================================================================

def main():
    np.random.seed(random_seed)
    svm_problem()
    
if __name__ == "__main__":
    main()