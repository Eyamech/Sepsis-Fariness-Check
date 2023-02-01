# -*- coding: utf-8 -*-


import streamlit as st



#EDA pkgs

import pandas as pd

import numpy as np



#data visualization pcks

import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('Agg')

import seaborn as sns

#import plotly.graph_objects as go

import plotly.express as px





#ML pckgs

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn import metrics



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

#disable warning message

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_option('deprecation.showPyplotGlobalUse', False)



import pickle



import base64

import time



timestr = time.strftime("%Y%m%d-%H%M%S")

# Fxn to Download Result

def download_link(object_to_download, download_filename, download_link_text):

    d=pickle.dump(lr_model, open(filename, 'wb'))

    # some strings <-> bytes conversions necessary here

    b64 = base64.b64encode(object_to_download.encode()).decode()

    

    return f'<a href="data:file/txt;base64,{b64}" download="{d}">"click click"</a>'



def make_downloadable(data):

    csvfile = data.to_csv(index=False)

    b64 = base64.b64encode(csvfile.encode()).decode()

    new_filename = "Analysis_report_{}_.csv".format(timestr)

    st.markdown("🏻  Download CSV file ⬇️  ")

    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click here!</a>'

    st.markdown(href, unsafe_allow_html=True)



def main():

    #"""Auto-Machine Learning App with Streamlit """

    st.title("Machine Learning Data Analytics")

   
  
    st.subheader("Hello everyone and welcome to my web application. As part of my graduation project, I am  going to explore different Classifiers to check fairness Trait in machine learning ")


             

   # st.text("~ Currently accepting smaller files due to Heroku free storage limit,thus computationally expensive tasks might fail and will restart the app, will be transfered to streamlit platform soon ")

  

  #  st.text("Ignore:EmptyDataError: No columns to parse from file have no effect on analysis,This is indentation Error will be fixed in next update")

   

    activities = ["Model Implementation"]

    

    choice = st.sidebar.selectbox("Select Activity",activities)

    

#--------------------------------- MODEL BUILDING---------------------------------------        

        

    if choice == 'Model Implementation':

        st.subheader("Create a Machine Learning Model")

        

        data = st.file_uploader("Upload Dataset",type = ["csv","txt"])

        if data is not None:

            df = pd.read_csv(data,sep=',')

            st.dataframe(df.head())
            st.dataframe(df.info())


            

        #if st.sidebar.checkbox("Show Dimensions(Shape)"):

         #   st.success("In (Row, Column) format")

          #  st.write(df.shape)

          

        if st.sidebar.checkbox("Missing Values"):
            st.subheader("Pre-Processing Step:")
            st.write("We start checking if our Data contains any missing values. We used isnull().sum to check how many missing values in each column")
            st.write(df.isnull().sum())
            st.write("The percentage of NaN values ")
            NaN_Values = df.isnull().sum() * 100 / len(df)
            st.write(NaN_Values)
            

        

        if st.sidebar.checkbox("Impute Missing Values*"):
            st.info("Features with more than 70% nan values will be deleted")
            drop_columns_70_nanvalues = NaN_Values[NaN_Values>70].keys()
            
            df = df.drop(drop_columns_70_nanvalues, axis=1)
            st.info("We impute the rest of the missing values using the median")

            #df = df.dropna()
            df.fillna(df["O2Sat"].median(),inplace=True)
            df.fillna(df["HospAdmTime"].median(),inplace=True)
            df.fillna(df["Resp"].median(),inplace=True)
            df.fillna(df["ICULOS"].median(),inplace=True)


            st.write(df.isnull().sum())

            

        if st.sidebar.checkbox("Remove Columns*"):
            st.write("In case we want to delete any column we can select it and the column will be deleted ")

            all_columns = df.columns.to_list()

            selected_columns = st.sidebar.multiselect("Select Columns like ID to remove",all_columns)

            st.write(str(selected_columns))

            new_drop_df = df.drop(selected_columns,axis=1)

            st.write(new_drop_df)

            

        if st.sidebar.checkbox("Define X and Y*"):
            st.write("Here we will define the independent variable as our target variable and the rest of columns as our dependent variables ")

            all_columns_names = new_drop_df.columns.tolist()

            #sca1, sca2 = st.beta_columns(2)

            #with sca1:

            if st.sidebar.checkbox("OR Selected X Variables"):

               X = st.sidebar.multiselect("Select Column for Independent Variables X",all_columns_names,key='dx')

               d_x = new_drop_df[X]

               st.info("X-Dependent Variables")

               st.write(d_x)

               

               Y1 = new_drop_df.drop(X,axis=1)              

               st.info("Y-Independent Variable")

               st.write(Y1)

               

            else:

               Y = st.sidebar.multiselect("Select Column for Dependent variable Y*",all_columns_names,key='dy')

               Y1 = new_drop_df[Y]

               st.info("Our Y-Inependent Variable is: ")

               st.write(Y1)

               

               d_x = new_drop_df.drop(Y,axis = 1)

               st.info("The X_Dependent Variables are:")

               st.write(d_x)

            

           

       

            

            

        if st.sidebar.checkbox("Select Columns to Use*",key='s01'):

            #st.info("Columns to Select")

            all_columns = d_x.columns.to_list()

            if st.sidebar.checkbox("Select X-var Columns",key='s02'):

               selected_columns = st.sidebar.multiselect("Select Columns",all_columns,key='s03')

               trans_df = d_x[selected_columns]

               st.success("X-Var Columns Selected")

               st.dataframe(trans_df)

            

            else:

               st.sidebar.checkbox("Or Select All Columns*",key='s04')

               all_columns = d_x.columns.to_list()

               #st.write(all_columns)

               trans_df = d_x[all_columns]

               st.success("Columns Selected")

               st.dataframe(trans_df)

               

        #if st.sidebar.checkbox("Transform Categorical variables*"):

            #all_columns = df.columns.to_list()

            #selected_columns = st.sidebar.multiselect("Select Columns like ID to transfrm",all_columns,key="trans01")

                

            #trans_df1 = pd.get_dummies(trans_df)

            #st.success("Transformation Successful")

            #st.write(trans_df1) 


        #if st.sidebar.checkbox("Scale/Normalize the Data*"):

            #sc = StandardScaler()

            #scaled_df = sc.fit_transform(trans_df1)

            #st.success("Scaled/Normalized Data")

            #st.write(scaled_df)

            #X_test = sc.transform(X_test)
        

            



        if st.sidebar.checkbox("Class Imbalance/Value counts"):
            st.write("We will check if the class attribute is balanced")

            st.warning("Class Imbalance Data for Y Dependent Variable")
            from collections import Counter
            st.write(Counter(Y1.SepsisLabel))
           

            

                        

#-----------------------GENERATE AI MODEL--------------------------------        

        

        if st.sidebar.checkbox("Generate AI Model"):

            

            #Metrics packages

            from sklearn import metrics

            from sklearn.metrics import classification_report, confusion_matrix

            from sklearn.model_selection import cross_val_score

            from sklearn.model_selection import KFold

            from sklearn.metrics import accuracy_score 

            from collections import Counter
            from imblearn.combine import SMOTETomek
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.over_sampling import SMOTE

            st.write("Before to start modelling, we splitt our data into train and test data ")
            X_train,X_test, y_train, y_test = train_test_split(df, Y1, train_size=0.7, random_state=0)

            

            #st.success("Class imbalanced Corrected for Train data")
            #all_columns = d_x.columns.to_list()
            Resampling = ["SMOTEtomek", "Over-sampling", "Under-sampling" ]
            if st.sidebar.multiselect("Select a resampling technique ",Resampling,key='dx'):

            #st.info("Columns to Select")

                all_columns = d_x.columns.to_list()

                if st.sidebar.checkbox("SMOTEtomek",key='s02'):

                    #selected_columns = st.sidebar.multiselect("Select Columns",all_columns,key='s03')
                    smt = SMOTETomek(random_state=42)

                    X_train, y_train = smt.fit_resample(X_train,y_train)

                    col_nam_y = y_train.columns.tolist()

                    st.write("After Over-Under Sampling, counts of Sepsis label 1 & 0", Counter(y_train["SepsisLabel"]))

                elif st.sidebar.checkbox("Over-sampling",key= 'dy'):
                
                    sm = SMOTE(sampling_strategy=1,random_state = 0)
                    
                    X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

                    col_nam_y = y_train.columns.tolist()
                    
                    st.write("After Over-Sampling, counts of Sepsis label 1 & 0", Counter(y_train["SepsisLabel"]))
                    
                elif st.sidebar.checkbox("Under-sampling",key= 's01'):
                    
                    undersample = RandomUnderSampler(sampling_strategy='majority')
                    X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
                    col_nam_y = y_train_under.columns.tolist()
                    st.write("After Under-Sampling, counts of Sepsis label 1 & 0", Counter(y_train_under["SepsisLabel"]))

                    #selected_columns = st.sidebar.multiselect("Select Columns",all_columns,key='s03')
                    
           

    

                #smt = SMOTETomek(random_state=42)

                #X_train, y_train = smt.fit_resample(X_train,y_train)

                #col_nam_y = y_train.columns.tolist()

                #st.write("After Over-Under Sampling, counts of Sepsis label 1 & 0", Counter(y_train["SepsisLabel"]))



            

            #algo = ['Logistic Regression','K-Nearest Neighbor(KNN)','Bayes’ Theorem: Naive Bayes','Linear Discriminant Analysis(LDA)','Linear Support Vector Machine(SVM)','Kernel Support Vector Machine(SVM)','SVM: with Paramter Tuning','Decision Tree: Rule-Based Prediction','Random Forest: Ensemble Method','eXtreme Gradient Boosting(XGBoost)']

            algo = ['K-Nearest Neighbor(KNN)','Linear Support Vector Machine(SVM)','Kernel Support Vector Machine(SVM)','SVM: with Paramter Tuning','Decision Tree: Rule-Based Prediction','Random Forest: Ensemble Method',]

            st.info("")

            st.success("")

            classifier = st.selectbox('Which algorithm?', algo)

            

            if classifier=='Decision Tree: Rule-Based Prediction':

                 dt_o1, dt_o2 = st.beta_columns(2)

                 with dt_o1:

                    dt_max_feat = st.select_slider("Select Features to consider when looking for the best split:",options=["auto","sqrt","log2","None"],value="auto",key='0svm34')

                    st.write("Default: None =  max_features=n_features.") 

                    dt_max_depth = st.slider("The Maximum depth of the tree",3,1000,500,key='0dtt33')

                    st.write("Default Max Limit =500 Taken:",dt_max_depth)

                    dt_min_sam_split = st.slider("Minimum number of samples required to split",2,10,2,key='0dttt133')

                    st.write("Default(min_samples_split)= 2, Taken:",dt_min_sam_split )

                  

                 with dt_o2:

                    dt_types = st.select_slider("Select Criterion( split type) ",options=["gini","entropy"],value=("gini"),key='0svm34')

                    st.write("Split Type:",dt_types) 

                    #OMITED for ERROR

                    #dt_min_impurity_split = st.slider("Impurity Split:Threshold for early stopping in tree growth.",0,5,0,key='0svmk33')

                    #st.write("Default(min_impurity_split)= 0 Taken", dt_min_impurity_split)

                    dt_min_sam_leaf = st.slider("Minimum number of sampels required to be a leaf node",1,10,1,key='0svmk33')

                    st.write("Default(min_samples_leaf)= 1, Taken:",dt_min_sam_leaf)

                    

                 dt_model = DecisionTreeClassifier(criterion = dt_types,max_depth=dt_max_depth, min_samples_split=dt_min_sam_split, min_samples_leaf=dt_min_sam_leaf, random_state = 0)

                 #FIXED OMITED min_impurity_split=dt_min_impurity_split

                 dt_model.fit(X_train, y_train)

                 if st.checkbox("Automate Best Optimal Parameters using Grid Search?"):

                     st.text(" Searching the Best Optimal Parameters,this will take few mins. as we crossvalidate 10times.However the results are biased and cannot gurantee exact accuracy due to randomised data structure pattern by GridSearch vs Decision Tree Classifier")

                     st.text("If clicked please wait until the process is complete else refresh the page")

                     parameters = [{'max_depth':[10, 100, 500, 1000], 'criterion': ['gini'],'min_samples_split':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5],'min_impurity_decrease':[0,1,2,3,4,5]},

                                   {'max_depth':[10, 100, 500, 1000], 'criterion': ['entropy'],'min_samples_split':[1,2,3,4,5,6,7,8,9,10],'min_samples_leaf':[1,2,3,4,5],'min_impurity_decrease':[0,1,2,3,4,5]}]

                     grid_search = RandomizedSearchCV(estimator = dt_model,param_distributions = parameters,scoring = 'accuracy',cv =10)

                     grid_search = grid_search.fit(X_train, y_train)

                     best_accuracy = grid_search.best_score_

                     best_parameters = grid_search.best_params_

                     st.write("The Best setting's score",(best_accuracy))

                     st.write(best_parameters)

                     st.info("Re-run the model with the settings above")  

                 else:

                     pass

                 acc = dt_model.score(X_test, y_test)

                 st.write('Accuracy: ', acc)

                 

                 y_pred_dt = dt_model.predict(X_test)

                 

                 st.success("Fairness Check")

                 cm=confusion_matrix(y_test,y_pred_dt)                 

                 st.write('Confusion matrix: ', cm)

                 st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,y_pred_dt))

                 st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test, y_pred_dt))

                 st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test, y_pred_dt))

                 st.write("F1 Score",metrics.f1_score(y_test, y_pred_dt))

                 st.write("Auc_roc score", metrics.roc_auc_score(y_test,y_pred_dt))

                 

                 st.success("Cross-Validation Fairness Check")

                 st.write("Cross Validation Fairness Check over 10 times")

                 kfold = KFold(n_splits=10)  

                 accuracies = cross_val_score(dt_model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                 st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                 st.write("Standard Deviation",accuracies.std())

                

                 # get importance

                 importance = dt_model.feature_importances_

                 st.info("Index column:Feature Numbers , Index Values:Score")

                 st.write(importance)

                # plot feature importance

                 st.success("Important Features Plot")

                 plt.bar([x for x in range(len(importance))], importance)

                 st.pyplot()

                 from sklearn.tree import export_text

                 st.info("Rules/Conditions used for prediction ")

                 col_names_dt=trans_df1.columns.tolist()

                 tree_rules = export_text(dt_model, feature_names=col_names_dt)

                 st.write(tree_rules)

                 st.info("Visualizing The Tree might be restricted due to HTML length & Width: will be updating soon")

                 st.info("")

                 st.success("")

                 st.warning("")

                

            if classifier=='Random Forest: Ensemble Method':

                 rf1, rf2 = st.beta_columns(2)

                 with rf1:

                    rf_n = st.slider("Select n_estimators/Tree",400,1500,500,key='0rf33')

                    st.write("Default: 500 Trees. Taken:", rf_n)

                    rf_min_sam_split = st.slider("Minimum number of samples required to split",2,10,2,key='0dttt133')

                    st.write("Default(min_samples_split)= 2, Taken:",rf_min_sam_split )

                    rf_max_feat = st.select_slider("Select Features to consider when looking for the best split:",options=["auto","sqrt","log2"],value="auto",key='0svm34')

                    st.write("Default: auto= sqrt(n_features) Taken:",rf_max_feat) 

                    

                 with rf2:

                    rf_p = st.select_slider("Select Criterion ",options=["entropy","gini"],value=("entropy"),key='0rf34')

                    st.write("Splitting Criteria:",rf_p)

                    rf_min_sam_leaf = st.slider("Minimum number of samples required to be a leaf node",1,10,1,key='0svmk33')

                    st.write("Default(min_samples_leaf)= 1, Taken:",rf_min_sam_leaf)

                    #OMITED for error

                    #rf_min_impurity_split = st.slider("Impurity Split:Threshold for early stopping in tree growth.",0,5,0,key='0svmk33')

                    #st.write("Default(min_impurity_split)= 0 Taken:",rf_min_impurity_split)

                    rf_oob = st.select_slider("Use out-of-bag samples to estimate the generalization accuracy?",options=["True","False"],value=("False"),key='0rf34')

                    st.write("Splitting Criteria:",rf_oob)

                    

                 #OMITED for ERROR min_impurity_split=rf_min_impurity_split,   

                 rf_model = RandomForestClassifier(n_estimators = rf_n, criterion = rf_p,min_samples_split=rf_min_sam_split, max_features=rf_max_feat,

                                                   min_samples_leaf=rf_min_sam_leaf, 

                                                   oob_score=rf_oob, random_state = 0)

                 rf_model.fit(X_train, y_train)

                 acc = rf_model.score(X_test, y_test)

                 st.write('Accuracy: ', acc)

                

                 y_pred_rf = rf_model.predict(X_test)

                 

                 st.success("Fairness Check")

                 cm=confusion_matrix(y_test,y_pred_rf)                 

                 st.write('Confusion matrix: ', cm)

                 st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,y_pred_rf))

                 st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test, y_pred_rf))

                 st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test, y_pred_rf))

                 st.write("F1 Score",metrics.f1_score(y_test, y_pred_rf))

                 st.write("Auc_roc score", metrics.roc_auc_score(y_test,y_pred_rf))

                 

                 st.success("Cross-Validation Fairness Check")

                 st.write("Cross Validation Fairness Check over 10 times")

                 kfold = KFold(n_splits=10)  

                 accuracies = cross_val_score(rf_model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                 st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                 st.write("Standard Deviation",accuracies.std())



                 # get importance

                 importance = rf_model.feature_importances_

                 st.info("Index column:Feature Numbers , Index Values:Score")

                 st.write(importance)

                 # plot feature importance

                 st.success("Important Features Plot")

                 plt.bar([x for x in range(len(importance))], importance)

                 st.pyplot()

                 st.info("")

                 st.success("")

                 st.warning("")

                        

            if classifier=='K-Nearest Neighbor(KNN)':

                 knn1, knn2 = st.beta_columns(2)

                 with knn1:

                    k_n = st.slider("Select N_neighbors",3,30,5,key='0k33')

                    st.write("N_neighbots (Default=5) Taken",k_n)

                    k_algo = st.select_slider("Algorithm used to compute the nearest neighbors:",options=["auto","ball_tree","kd_tree","brute"],value=("auto"),key='0rf34')

                    st.write("Default(algorithm):Auto, Taken",k_algo) 

                    

                 with knn2:

                    k_p = st.slider("Select Distance metrics: 1 is equivalent to the Manhattan distance and 2 is equivalent to the Euclidean distance ",1,2,2,key='0k34')

                    k_leaf = st.slider("Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query",10,70,30,key='0k33')

                    st.write("Default(leaf_size):30 Taken:",k_leaf)

                    k_weight = st.select_slider("Select weight method for all points in each neighborhood",options=["uniform","distance"],value=("uniform"),key='0rf34')

                    st.write("Default(weights):Auto, Taken:",k_weight)

                                     

                 knn_model = KNeighborsClassifier(n_neighbors = k_n,weights=k_weight, p = k_p, algorithm=k_algo, leaf_size=k_leaf)

                 knn_model.fit(X_train, y_train)

                 

                 if st.checkbox("Automate Best Optimal Parameters using Grid Search?"):

                     st.text(" Searching the Best Optimal Parameters,this will take few mins. as we crossvalidate several times. However the results are biased and cannot gurantee exact accuracy due to randomised data structure pattern taken by GridSearch vs KNN-Classifier")

                     st.text("If clicked please wait until the process is complete else refresh the page")

                     clf= KNeighborsClassifier()

  

                     parameters1 = [{'n_neighbors': [3, 5, 10,15,20,25], 'algorithm': ['auto','ball_tree','kd_tree','brute'],'leaf_size':[20,30,40,50,60,70,80,90]}]

                     grid_search = RandomizedSearchCV(estimator =clf,param_distributions = parameters1,scoring = 'accuracy',cv=2)

                     grid_search = grid_search.fit(X_train, y_train)

                     best_accuracy = grid_search.best_score_

                     best_parameters = grid_search.best_params_

                    

                     st.write("The Best setting's score",(best_accuracy))

                     st.write(best_parameters)

                     st.info("Re-run the model with the settings above")  

                 else:

                     pass                     

                                  

                 acc = knn_model.score(X_test, y_test)

                 st.write('Accuracy:', acc)

                 # Predicting the Test set results

                 y_pred_knn = knn_model.predict(X_test)

                 

                 

                 st.success("Fairness Check")

                 cm=confusion_matrix(y_test,y_pred_knn)                 

                 st.write('Confusion matrix: ', cm)

                 st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,y_pred_knn))

                 st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test, y_pred_knn))

                 st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test, y_pred_knn))

                 st.write("F1 Score",metrics.f1_score(y_test, y_pred_knn))

                 st.write("Auc_roc score", metrics.roc_auc_score(y_test,y_pred_knn))

                 

                 st.success("Cross-Validation Fairness Check")

                 st.write("Cross Validation Fairness Check over 10 times")

                 kfold = KFold(n_splits=10)  

                 accuracies = cross_val_score(knn_model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                 st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                 st.write("Standard Deviation",accuracies.std())

                 

                 st.error("Try another Algorithm for Feature importance")

                 st.info("")

                 st.success("")

                 st.warning("")

                 

            

                  



                

            if classifier == 'Linear Support Vector Machine(SVM)':

                svm_model=SVC(kernel = 'linear', random_state = 0)

                svm_model.fit(X_train, y_train)

                acc = svm_model.score(X_test, y_test)

                st.write('Accuracy: ', acc)

                svm_pred = svm_model.predict(X_test)

                

                st.success("Fairness Check")

                cm=confusion_matrix(y_test,svm_pred)                 

                st.write('Confusion matrix: ', cm)

                st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,svm_pred))

                st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test,svm_pred))

                st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test,svm_pred))

                st.write("F1 Score",metrics.f1_score(y_test,svm_pred))

                st.write("Auc_roc score", metrics.roc_auc_score(y_test,svm_pred))

                 

                st.success("Cross-Validation Fairness Check")

                st.write("Cross Validation Fairness Check over 10 times")

                kfold = KFold(n_splits=10)  

                accuracies = cross_val_score(svm_model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                st.write("Standard Deviation",accuracies.std())

                

                

                

                # get importance

                importance = svm_model.coef_[0]

                st.info("Index column:Feature Numbers , Index Values:Score")

                st.write(importance)

                # plot feature importance

                st.success("Important Features Plot")

                plt.bar([x for x in range(len(importance))], importance)

                st.pyplot()

                st.info("")

                st.success("")

                st.warning("")

                

                

            if classifier == 'Kernel Support Vector Machine(SVM)':

                ksvm_model=SVC(kernel = 'rbf', random_state = 0)

                ksvm_model.fit(X_train, y_train)

                acc = ksvm_model.score(X_test, y_test)

                st.write('Accuracy: ', acc)

                ksvm_pred = ksvm_model.predict(X_test)

                

                st.success("Fairness Check")

                cm=confusion_matrix(y_test,ksvm_pred)                 

                st.write('Confusion matrix: ', cm)

                st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,ksvm_pred))

                st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test,ksvm_pred))

                st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test,ksvm_pred))

                st.write("F1 Score",metrics.f1_score(y_test,ksvm_pred))

                st.write("Auc_roc score", metrics.roc_auc_score(y_test,ksvm_pred))

                 

                st.success("Cross-Validation Fairness Check")

                st.write("Cross Validation Fairness Check over 10 times")

                kfold = KFold(n_splits=10)  

                accuracies = cross_val_score(ksvm_model,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                st.write("Standard Deviation",accuracies.std())

                

                st.error("Try another Algorithm for Feature importance")

                st.info("")

                st.success("")

                st.warning("")

                

            if classifier=='SVM: with Paramter Tuning':

                 svm_o1, svm_o2 = st.beta_columns(2)

                 with svm_o1:

                    svm_gamma = st.select_slider("Select Gamma(gamma) ",options=["scale","auto"],value=("scale"),key='0svm34')

                    st.write("Kernel Gamma:",svm_gamma) 

                    svm_c = st.slider("Select Regularization parameter (C)",100,1000,1,key='0svmk33')

                    st.write("Default: 1.0")

                    svm_iter = st.slider("Select Iterations",1,1000,-1,key='0svmk33')

                    st.write("Default: -1 = No limit")

                    

                 with svm_o2:

                    svm_types = st.select_slider("Select Criterion(kernel type) ",options=["poly","rbf","sigmoid","linear"],value=("sigmoid"),key='0svm34')

                    st.write("Kernel Type:",svm_types) 

                    svm_degree = st.slider("Select degree for poly kernel(Ignored by all other Kernels)",2,8,3,key='0svmk33')

                    st.write("Default: 3")

           

                 ksvm_model2=SVC(kernel = svm_types,gamma=svm_gamma,C=svm_c,max_iter = svm_iter,degree=svm_degree,random_state = 0)

                 ksvm_model2.fit(X_train, y_train)                 

                 if st.checkbox("Automate Best Optimal Parameters using Grid Search?"):

                     st.text(" Searching the Best Optimal Parameters,this will take few mins. as we crossvalidate 3times. However the results are biased and cannot gurantee exact accuracy due to randomised data structure pattern by GridSearch vs SVM Classifier")

                     st.text("If clicked please wait until the process is complete else refresh the page")

                     parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

                                   {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

                     grid_search = RandomizedSearchCV(estimator = ksvm_model2,param_distributions= parameters,scoring = 'accuracy',cv = 3)

                     grid_search = grid_search.fit(X_train, y_train)

                     best_accuracy = grid_search.best_score_

                     best_parameters = grid_search.best_params_

                     st.write("The Best setting's score-",(best_accuracy))

                     st.write(best_parameters)

                     st.info("Re-run the model with the settings above")  

                 else:

                     pass                                                                 

                 acc = ksvm_model2.score(X_test, y_test)

                 st.write('Accuracy: ', acc)

                 ksvm_pred2 = ksvm_model2.predict(X_test)

                 

                 st.success("Fairness Check")

                 cm=confusion_matrix(y_test,ksvm_pred2)                 

                 st.write('Confusion matrix: ', cm)

                 st.write('Balanced Accuracy ',metrics.balanced_accuracy_score(y_test,ksvm_pred2))

                 st.write("Recall Accuracy Score~TP",metrics.recall_score(y_test,ksvm_pred2))

                 st.write("Precision Score ~ Ratio of TP",metrics.precision_score(y_test,ksvm_pred2))

                 st.write("F1 Score",metrics.f1_score(y_test,ksvm_pred2))

                 st.write("Auc_roc score", metrics.roc_auc_score(y_test,ksvm_pred2))

                 

                 st.success("Cross-Validation Fairness Check")

                 st.write("Cross Validation Fairness Check over 10 times")

                 kfold = KFold(n_splits=3)  

                 accuracies = cross_val_score(ksvm_model2,X= X_train,y= y_train,cv = kfold,scoring='accuracy')

                 st.write("Mean/Avergae Accuracy Score",accuracies.mean())

                 st.write("Standard Deviation",accuracies.std())

                 

                 cm=confusion_matrix(y_test,ksvm_pred2)

                 st.write('Confusion matrix: ', cm)

                 st.error("Try another Algorithm for Feature importance")

                 st.info("")

                 st.success("")

                 st.warning("")

                    

                                

                

                

                

           



#-----------------------------------PREDICTION---------------------------------

    

        if st.sidebar.checkbox("Predict"):

            data = st.file_uploader("Upload your data you want to predict",type = ["csv","txt"])

            if data is not None:

                df_temp = pd.read_csv(data,sep=',')

                df_temp = df_temp.dropna()

                st.success("Removed Missing Values")

                st.dataframe(df_temp.head())

                st.info("Mapping the number of the variables from the model")

                st.write("Found columns", len(trans_df.columns))

                

            fr_new_columns = trans_df.columns.to_list()

               #st.write(all_columns)

            trans_newdf = df_temp[fr_new_columns]

            st.success("Columns Mapped")

            st.dataframe(trans_newdf)

            st.write("Mapped columns", len(trans_newdf.columns))

             

            ddd_df1 = pd.get_dummies(trans_newdf)

            st.success("Transformation Successful")

            st.write(ddd_df1) 

                    

            sc = StandardScaler()

            scaled_df1 = sc.fit_transform(ddd_df1)

            st.success("Scaled/Normalized Data")

            st.write(scaled_df1)

            

            #---------------- PREDICT NEW DATA using SELECTED MODEL---------

            

            #st.write(algo) getting the list of model from above

            #algo = ['Logistic Regression','K-Nearest Neighbor(KNN)','Bayes’ Theorem: Naive Bayes','Linear Discriminant Analysis(LDA)','Linear Support Vector Machine(SVM)','Kernel Support Vector Machine(SVM)','SVM: with Paramter Tuning','Decision Tree: Rule-Based Prediction','Random Forest: Ensemble Method','eXtreme Gradient Boosting(XGBoost)']

            selected_model_names = st.selectbox("Select Models To Predict Your New Data",algo)

            st.text(" Note: must use respective Generate Ai model before predicting")

            if selected_model_names =='Logistic Regression':

                results = lr_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)

                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='K-Nearest Neighbor(KNN)':

                results = knn_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

            

            

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='Linear Support Vector Machine(SVM)':

                results = svm_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable column as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='Kernel Support Vector Machine(SVM)':

                results = ksvm_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='SVM: with Paramter Tuning':

                results = ksvm_model2.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='Decision Tree: Rule-Based Prediction':

                results = dt_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                

                 #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            elif selected_model_names=='Random Forest: Ensemble Method':

                results = rf_model.predict(scaled_df1)

                st.success("Predicted Results")

                st.write(results)

                #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

                

            

                 

            #------------Putting All Together The Results--------------------------

                inv_df = sc.inverse_transform(scaled_df1)



                #need to convert to pandas else will throw error 'col' not found as it a numpy object

                inv_df2  = pd.DataFrame(inv_df)

                #st.write(inv_df2)

                       

                #geting the column name as list

                c_names = ddd_df1.columns.tolist()

            

                #adding column names to the dataset                        

                inv_df2.columns=c_names

                #st.write(inv_df2)

                st.success("Putting all together your uploaded data and the predicted results")

            

                #concat--------

                #st.write(Y1.columns)

                #need to convert to pandas else numpy cant find columns error

                y_col = pd.DataFrame(Y1.columns)

                #st.write(y_col)

                #getting the column name as list

                y_names = y_col.values.tolist()

                #st.write(y_names)



                #again convert to pandas Dataframe else throw numpy error

                results2=pd.DataFrame(results)

                #renaming/labeling the Y column          

                results2.columns=y_names

                #st.write(results2)

       

                #series not applicable columna as name s = pd.Series(results2)

                p_results = pd.concat([inv_df2,results2],axis=1)

                #working:p_results = pd.concat([inv_df2,s],axis=1)

                st.write(p_results)

       #--------------- THE END---------------------------     

         

                                



        





if __name__ == '__main__':

    main()