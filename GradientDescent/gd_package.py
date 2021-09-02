from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
import math
import numpy as np
import matplotlib.pyplot as plt

class EvaluationMetrics:
    
    def __init__(self):
        pass
    
    def calculate_metrics(self, metric, y_pred, y):
        
        func_mapper = {
            'rmse' : self.rmse(y_pred, y),
            'accuracy' : self.accuracy_score(y_pred, y),
            'precision' : self.precision_score(y_pred, y),
            'recall' : self.recall_score(y_pred, y)
        }
        
        return func_mapper[metric]
    
    def rmse(self, y_pred, y):
        return math.sqrt(mean_squared_error(y, y_pred))
    
    def accuracy_score(self, y_pred, y):
        return accuracy_score(y, y_pred)
    
    def precision_score(self, y_pred, y):
        return precision_score(y, y_pred)
    
    def recall_score(self, y_pred, y):
        return recall_score(y, y_pred)
        

class Visualize:
    
    def __init__(self):
        pass
    
    def visualize_plot(self, fig_details, x_label, y_label, plot_title):
        fig, ax = plt.subplots()
        for fig_ in fig_details:
            ax.plot(fig_[0], fig_[1], label = fig_[2])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(plot_title)
        ax.legend()
    
class LinearRegression:
    
    def __init__(self, num_of_samples):
        self.num_of_samples = num_of_samples
        self.evalMetObj = EvaluationMetrics()
        self.visObj = Visualize()
        self.train_rmse = []
        self.test_rmse = []
        self.samples_list = []
        self.cost_history = []
        self.iterator = 0
        
    def get_predictions(self, X, theta):
        return X.dot(theta)
        
    def calculate_gradients(self, alpha_lr, theta, X, y_pred, y):
        delta = (1/self.num_of_samples) * (X.T.dot(y_pred - y))
        theta = theta - (alpha_lr * delta)
        return theta
    
    def calculate_cost(self, y_pred, y):
        self.cost_history.append((1/2*self.num_of_samples) * np.sum(np.square(y_pred - y)))
    
    def evaluate_model(self, y_train_pred, y_train, y_test_pred, y_test):
        self.train_rmse.append(self.evalMetObj.calculate_metrics('rmse', y_train_pred, y_train))
        self.test_rmse.append(self.evalMetObj.calculate_metrics('rmse', y_test_pred, y_test))
        self.samples_list.append(self.iterator + 1)
        self.iterator += 1
    
    def display_metrics(self):
        
        #Draw training cost plot
        self.visObj.visualize_plot([(self.samples_list, self.cost_history, 'Train Cost')], 'Iterations', 'Cost', 'Cost Plot')
        
        #Print Train and Test RMSE after N iterations
        print(f"Final Train RMSE : {self.train_rmse[len(self.train_rmse) - 1]}")
        print(f"Final Test RMSE : {self.test_rmse[len(self.test_rmse) - 1]}")
        self.visObj.visualize_plot([(self.samples_list, self.train_rmse, 'Train RMSE'), 
                                    (self.samples_list, self.test_rmse, 'Test RMSE')], 'Iterations', 'RMSE', 'RMSE Plot')
        
    
class LogisticRegression:
    
    def __init__(self, num_of_samples, num_of_features):
        self.num_of_samples = num_of_samples
        self.num_of_features = num_of_features
        self.w = np.zeros((1, self.num_of_features))
        self.b = np.zeros((1, 1))
        self.samples_list = []
        self.cost_history = []
        self.train_acc = None
        self.test_acc = None
        self.precision_score = None
        self.recall_score = None
        self.iterator = 0
        self.evalMetObj = EvaluationMetrics()
        self.visObj = Visualize()
        
    def get_predictions(self, X):
        z = self.w.dot(X.T) + self.b #1 * n_features | m_samples * n_features -> 1 * m_samples
        pred = (1 / (1 + (1 / np.exp(z))))
        return pred.T #m_samples * 1
        
    def calculate_gradients(self, alpha_lr, X, y_pred, y):
        dz = y_pred - y #m_samples * 1
        dw = (1 / self.num_of_samples) * (X.T.dot(dz)) #m_samples * n_features | m_samples * 1 -> n_features * 1
        db = (1 / self.num_of_samples) * (np.sum(dz)) #1 * 1
        self.w = self.w - (alpha_lr * dw.T)
        self.b = self.b - (alpha_lr * db)
        
    def calculate_cost(self, y_pred, y):
        self.cost_history.append((1 / self.num_of_samples) * np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))
        self.samples_list.append(self.iterator + 1)
        self.iterator += 1
    
    def convert_predictions_to_classes(self, y_pred, threshold = 0.5, classes = 2):
        y_pred = y_pred.flatten()
        y_pred_class = []
        for i in range(0, len(y_pred)):
            y_class = 0 if y_pred[i] < threshold else 1
            y_pred_class.append(y_class)
            
        return y_pred_class
    
    def evaluate_model(self, y_train_pred, y_train, y_test_pred, y_test):
        self.train_acc = self.evalMetObj.calculate_metrics('accuracy', y_train_pred, y_train)
        self.test_acc = self.evalMetObj.calculate_metrics('accuracy', y_test_pred, y_test)
        self.precision_score = self.evalMetObj.calculate_metrics('precision', y_test_pred, y_test)
        self.recall_score = self.evalMetObj.calculate_metrics('recall', y_test_pred, y_test)
        
    def display_metrics(self):
        
        print(f"Train Accuracy : {self.train_acc}")
        print(f"Test Accuracy : {self.test_acc}")
        print("-----------------------------------------")
        
        print(f"Precision Score (Test Set) : {self.precision_score}")
        print("Of All Values where the model predicted 1, what fraction of them were actually 1 [TP / TP + FP]")
        print("-----------------------------------------")
        print(f"Recall Score (Test Set) : {self.recall_score}")
        print("Of All Values that are 1, what fraction of them did the model correctly predict as 1 [TP / TP + FN]")
        print("-----------------------------------------")
        
        #Draw training cost plot
        self.visObj.visualize_plot([(self.samples_list, self.cost_history, 'Train Cost')], 'Iterations', 'Cost', 'Cost Plot')