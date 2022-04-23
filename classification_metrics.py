class ModelMetricsCalculator():
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
        self._model = classifier # model to predict
        self._model.trainable = False # freeze layer weights
        self._X = X # input data
        self._Y = Y # predictions
        self._Make_predictions() # perform predictions (probability and label)
        self.confusion_matrix = metrics.confusion_matrix(self._Y, self._labelPredictions)
        
    # Predict classifier outputs
    def _Make_predictions(self):
        self._probabilityPredictions = model.predict(self._X)
        self._labelPredictions = tf.math.argmax(self._probabilityPredictions, axis=1).numpy()