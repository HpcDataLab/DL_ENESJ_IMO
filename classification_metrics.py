class MetricsCalculator():
    def __init__(self, Y_labels, Y_probabilityPreds, labels_names=None):
        import numpy as np
        import sklearn.metrics as metrics
        self._Y = Y_labels # correct labels
        self._probabilityPredictions = Y_probabilityPreds # predictions with probability distribution
        self._unique_labels = np.unique(self._Y)
        self._labelPredictions = tf.math.argmax(self._probabilityPredictions, axis=1).numpy()
        self._confusion_matrix = metrics.confusion_matrix(self._Y, self._labelPredictions)
        self._withFigures = False # if figures will be generated
        self._figuresPath = None # non given path
        self.labels_names = labels_names
    
    
    # If is set, images will be generated
    def SetFiguresOn(self, path=''):
        self._figuresPath = path
        self._withFigures = True
        
    def SetFiguresOff(self):
        self._withFigures = False
    
    
    # Compute confusion matrix for specific label if given
    def Confusion_matrix(self, target_label_index=None, return_type="matrix"):
        """Computes a binary confusion matrix for `target_label` if given.
        Otherwise returns the multi-class confusion matrix.
        Parameters
        ----------
        target_label_index : int, default=None
            Label index whom to compare with other classes 
            to perform confusion matrix.
        return_type : str, default="matrix"
            If binary (or target_label_index is specified) classification problem, 
            determines the return format. Acceptable values are "matrix" and "dict".
            
            If "matrix" is given, following return format will be followed:
            
                     labels
                p  -----------
                r  | TP | FP |
                e  -----------
                d  | FN | TN |
                s  -----------
            
            Otherwise a dictionary with the corresponding keys will be returned.
        Returns
        -------
        confusion_matrix : array-like, dict
            Confusion matrix from labels with TP, TN, FP and TN values.
        """
        import sklearn.metrics as metrics
        cm = self._confusion_matrix
        nLabels = len(self._unique_labels) # number of labels to predict
        if return_type not in ["matrix", "dict"]:
            raise ValueError(f'{return_type} is an invalid `return_type` value. Acceptable values are "matrix" and "dict".')
        
        # Save heatmap
        if self._withFigures:
            from os import path, mkdir
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            ax = sns.heatmap(cm, annot=True, cmap='Blues')
            ax.set_title('Confusion Matrix\n\n');
            ax.set_xlabel('\nActual Values');
            ax.set_ylabel('Predicted Values ');

            if self.labels_names:
                ## Ticket labels - List must be in alphabetical order
                ax.xaxis.set_ticklabels(self.labels_names)
                ax.yaxis.set_ticklabels(self.labels_names)

            # Create directory if it does not exist yet
            if not path.isdir(self._figuresPath):
                mkdir(self._figuresPath)
            ## Display the visualization of the Confusion Matrix
            plt.savefig(f"{os.path.join(self._figuresPath, 'confusionMatrix.png')}", dpi=600)
            plt.cla() # Clear axes
        
        # If not particular label, return complete confusion matrix
        if target_label_index == None:
            # return sklearn matrix
            if return_type == "matrix":
                return cm
            # create dict from values
            elif return_type == "dict" and nLabels == 2:
                return {
                    "TP": cm[0][0],
                    "TN": cm[0][1],
                    "FP": cm[1][0],
                    "TN": cm[1][1]
                }
        # if invalid input fiven
        if target_label_index < 0 or type(target_label_index) != int:
            raise ValueError(f"`target_label_index` must be a positive integer value, {target_label_index} given")
        if target_label_index > nLabels:
            raise ValueError(f"Index {target_label_index} is greater that the available number of classes")
        
        i = target_label_index # rename index of desired label
        TP = cm[i][i] # correct label predictions
        FP = np.sum(cm[i]) - TP # incorrect label predictions
        FN = np.sum(cm, axis=0)[i] - TP # incorrect other labels predictions
        TN = np.sum(cm) - TP - FP - FN # correct other labels predictions
        
        if return_type == "matrix":
            return [[TP, FP], [FN, TN]]
        elif return_type == "dict":
            return {
                "TP": TP,
                "FN": FN,
                "FP": FP,
                "TN": TN
            }
    
    # Compute accuracy over results
    def Accuracy(self, target_label_index=None, confusion_matrix=None, confusion_dict=None):
        """Measures accuracy.
        Parameters
        ----------
        target_label_index : int, default=None
            Label index whom to compare with other classes 
            to perform confusion matrix.
        confusion_matrix (optional) : array-like
            Confusion matrix where to take model performance and
            measure accuracy.
        Returns
        -------
        accuracy : float
            Accuracy measure.
        """
        import sklearn.metrics as metrics
        # If not confusion matrix was given, compute it from scratch
        if not confusion_matrix or confusion_dict:
            confusion_matrix = self.Confusion_matrix(return_type="matrix") if not target_label_index else self.Confusion_matrix(return_type="matrix", target_label_index=target_label_index)
        # If confusion matrix is given, convert into dictionary
        if confusion_dict:
            confusion_matrix = [[confusion_dict["TP"], confusion_dict["FP"]], [confusion_dict["FN"], confusion_dict["TN"]]]
        return np.sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))]) / np.sum(confusion_matrix) # compute overall correct predictions
    
    
    # Measure unbalanced metrics
    def _UnbalancedMetric(self, metric, options=None):
        if options:
            r = options
        else:
            r = {"micro": None, "macro": None, "weighted": None}
        for k in r.keys():
            r[k] = metric(self._Y, self._labelPredictions, average=k)
        return r
    
    
    # Measures the AUC of the given index
    def AUC(self):
        raise NotImplementedError("TODO")
        
    
    # Measures the IoU of the given index
    def IoU(self):
        import sklearn.metrics as metrics
        return self._UnbalancedMetric(metrics.jaccard_score)
        
        
    # Compute recall (sensitivity)
    def Recall(self):
        import sklearn.metrics as metrics
        return self._UnbalancedMetric(metrics.recall_score)
    
    
    # Compute precision
    def Precision(self):
        import sklearn.metrics as metrics
        return self._UnbalancedMetric(metrics.precision_score)
    

    # Compute F1-Score
    def F1score(self):
        import sklearn.metrics as metrics
        return self._UnbalancedMetric(metrics.f1_score)
    
    
    # Compute CohenKappa
    def CohenKappa(self):
        import sklearn.metrics as metrics
        return metrics.cohen_kappa_score(self._Y, self._labelPredictions)
    
    
    # Compute and display ROC Curves
    def ROCCurve(self, type_multi_class="ovo"):
        import sklearn.metrics as metrics
        return roc_auc_score(self._Y, self._probabilityPredictions, multi_class=type_multi_class)
    
    
    # Compute all metrics
    def MetricsReport(self):
        metrics_options = {
            "accuracy": self.Accuracy, 
            "IoU": self.IoU, 
            "recall": self.Recall, 
            "precision": self.Precision, 
            "f1-score": self.F1score, 
            "CohenKappa": self.CohenKappa,
        }
        r = dict()
        for metric, method in metrics_options.items():
            r[metric] = method()
        return r
    
        
    
    ##### AUC
    ##### IoU
    ##### Recall (sensitivity)
    ##### Precision
    ##### Specificity
    ##### F1 Score
    ##### CohenKappa
    ##### ROC Curve figure
    ##### Confusion matrix (figure)


class ModelMetricsCalculator(MetricsCalculator):
    """Compute most common performance metrics for a binary
    or a multiple classification model.
    Parameters
    ----------
    classifier : tensorflow classification model
        Trained classifier model to be used for predictions.
    X : array-like, or tensor
        Inputs to be passed to the classifier.
    Y : array-like, or tensor
        Target labels to be predicted by the classifier.
    """
    
    def __init__(self, classifier, X, Y, labels_names=None):
        import sklearn.metrics as metrics
        import numpy as np
        classifier.trainable = False # freeze layer weights
        probabilityPredictions = model.predict(X)
        super().__init__(Y, probabilityPredictions, labels_names) # init from parent class