import numpy as np
import fit_the_thing as f

class CreditModel:
    def __init__(self):
        """
        Instantiates the model object, creating class variables if needed.
        """

        # TODO: Initialize your model object.
        pass

    def fit(self, X_train, y_train):
        """
        Fits the model based on the given `X_train` and `y_train`.

        You should somehow manipulate and store this data to your model class
        so that you can make predictions on new testing data later on.
        """
        
        self.param = f.model(X_train,  y_train)
        pass

    def predict(self, X_test):
        """
        Returns `y_hat`, a prediction for a given `X_test` after fitting.

        You should make use of the data that you stored/computed in the
        fitting phase to make your prediction on this new testing data.
        """
        return f.predict(self.param,X_test)
        # TODO: Predict on `X_test` based on what you learned in the fit phase.

