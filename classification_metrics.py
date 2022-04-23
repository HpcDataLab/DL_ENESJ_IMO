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
    
    def __init__(self, classifier, X, Y):
        import sklearn.metrics as metrics
        import numpy as np
        self._model = classifier # model to predict
        self._model.trainable = False # freeze layer weights
        self._X = X # input data
        self._Y = Y # predictions
        self._Make_predictions() # perform predictions (probability and label)
        self._confusion_matrix = metrics.confusion_matrix(self._Y, self._labelPredictions)
        self._unique_labels = np.unique(self._Y)
        
    # Predict classifier outputs
    def _Make_predictions(self):
        self._probabilityPredictions = model.predict(self._X)
        self._labelPredictions = tf.math.argmax(self._probabilityPredictions, axis=1).numpy()
    

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
        cm = self._confusion_matrix
        nLabels = len(self._unique_labels) # number of labels to predict
        if return_type not in ["matrix", "dict"]:
            raise ValueError(f'{return_type} is an invalid `return_type` value. Acceptable values are "matrix" and "dict".')
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