To load the model use joblib library
eg.
loaded_rf = joblib.load('filepath')

Use this method for prediction
For eg.
loaded_rf.predict(X_test_norm)
