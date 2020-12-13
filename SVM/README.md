# This folder contains the code for the SVM model 

-   `svm.ipynb`:
    -   The jupyter notebook file used to train SVM model

-   `svm.joblib`:
    -   SVM Model model saved after training it with genres as features
    -   To load and run the model:
        -   ` loaded_svm = joblib.load("svm.joblib")`
        -   `loaded_svm.predict(X_test_norm)`

        
