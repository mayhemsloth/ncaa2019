'''
@author: Thomas
Created to train the logistic regression, neural network, or 
support vector machine model of the .csv data file that was exported by my other Python scripts.
'''

import numpy as np
import pandas as pd
import scipy as sp
import sklearn as skl
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier

#Load the data from the csv to a dataframe in pandas
training_df_name = '../src/training_data.csv'
training_data_df = pd.read_csv(training_df_name)
#Load the submission setup data from the csv to a dataframe in pandas
submission_name = '../src/submission_setup_Stage2.csv'
submission_df = pd.read_csv(submission_name)

y = training_data_df['Result']  #sets the result vector y

'''
Choose your columns here from the list/permutations of
        Team1_+      Team2_+     delta+      (concatenated)
                                        of any of the following
Seed   mOrd    Wins   Losses  PointsScored    FGM  FGA   FGM3  FGA3   FTM   FTA  OR  DR
AST  TO   Stl   Blk   PF   NumOT   WinPercent   FGPercent  FTPercent
'''
all_delta_cols = ['deltaSeed', 'deltamOrd', 'deltaWins', 'deltaLosses', 'deltaPointsScored', 'deltaFGM', 'deltaFGA', 'deltaFGM3', 'deltaFGA3', \
                  'deltaFTM', 'deltaFTA', 'deltaOR', 'deltaDR', 'deltaAST', 'deltaTO', 'deltaStl', 'deltaBlk', 'deltaPF', 'deltaNumOT', \
                  'deltaWinPercent', 'deltaFGPercent', 'deltaFTPercent']
cols_to_train = ['deltaSeed', 'deltaPointsScored', 'deltaWins', 'deltaFGPercent','deltaOR','deltaWinPercent']   #chosen columns to train on if using this



X = training_data_df[all_delta_cols]        #creates the dataframe for training the data in Logistic Regression
X_submission = submission_df[all_delta_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #split the data set into 70/30 split for cross validation of C values


#Scaling the data before giving it to LogisticRegression
scaler = StandardScaler()               #initializes the scaler variable
X_scaled = scaler.fit_transform(X_train,y_train)    #calculates the mean data, and then transforms it all
X_test_scaled = scaler.transform(X_test)            #transforms the test data with the same mean/variance as before
X_submission_scaled = scaler.transform(X_submission)      #transforms the submission data with the same mean/variance as training set data



#NOTE: When training with ALL delta cols, C=0.026 is the lowest logloss at 0.587916 using a cross validation test set
C_numbers = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]           #various C values to try to investigate
C_numbers_fine = range(1,100) #fine-tuning the exact C value
lowest_logloss = 1.0  #helper variable for grabbing info out of for loop
C_lowest_logloss = 0.0 #helper variable for grabbing info out of for loop

''' THIS IS CODE USED TO HELP TRAIN MULTIPLE MODELS AND PICK OUT THE BEST C VALUE FOR LOGISTIC REGRESSION
for C_num in C_numbers:
    modelLR = LogisticRegression(C=C_num).fit(X_scaled,y_train)     #creates the linear regression classification model
    predicted_back_prob = modelLR.predict_proba(X_test_scaled)   #gets back the predicted probabilities of the training set data
    predicted_back = modelLR.predict(X_test_scaled)             #gets back the predicted outcome (y vals) of the training set data
    
    logloss = metrics.log_loss(y_test,predicted_back_prob)
    print('Statistics for ', C_num)
    print('Log Loss:', metrics.log_loss(y_test,predicted_back_prob))
    print('Accuracy:', modelLR.score(X_test_scaled,y_test))
    if logloss < lowest_logloss:
        lowest_logloss = logloss
        C_lowest_logloss = C_num
    #print('Classification Report:\n', metrics.classification_report(y_test,predicted_back))
print('C = ' + str(C_lowest_logloss) + ' with logloss = ' + str(lowest_logloss))
'''


''' LOGISTIC REGRESSION
modelLR = LogisticRegression(C=0.026).fit(X_scaled,y_train)
predicted_sub_prob = modelLR.predict_proba(X_submission_scaled)

submission_df['Pred'] = predicted_sub_prob[:,1]    #the predicted_prob returns a two column matrix, where the second column gives the positive (y=1) probability
submission_df[['ID', 'Pred']].to_csv(r'C:\\users\\Thomas\\workspace\\mm_ncaa2019\\src\\submission_scikit_LR_alldeltacols_Ceqp026.csv', index=False)

print('Log Loss:', metrics.log_loss(y_test, modelLR.predict_proba(X_test_scaled)))
print('Accuracy:', modelLR.score(X_test_scaled,y_test))

modelLR_coeff_df = pd.DataFrame(zip(X.columns, np.transpose(modelLR.coef_)))
print('Model Coefficient:', modelLR_coeff_df)
'''
num_range = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
for num in num_range:
    NN = MLPClassifier(activation = 'logistic', solver='lbfgs', alpha=num, hidden_layer_sizes=(900,900), random_state=1).fit(X_scaled,y_train)
    NN_back_prob = NN.predict_proba(X_test_scaled)
    NN_back_prob_TRAIN = NN.predict_proba(X_scaled) 
    print(num, 'Log Loss test:', metrics.log_loss(y_test,NN_back_prob), 'Score test:', NN.score(X_test_scaled,y_test))
    print(num, 'Log Loss trai:', metrics.log_loss(y_train,NN_back_prob_TRAIN), 'Score trai:', NN.score(X_scaled,y_train))
    print(' ')

NN = MLPClassifier(activation = 'logistic', solver='lbfgs', alpha=7, hidden_layer_sizes=(900), random_state=1).fit(X_scaled,y_train)
predicted_sub_probNN = NN.predict_proba(X_submission_scaled)
#submission_df['Pred'] = predicted_sub_probNN[:,1] 
#submission_df[['ID', 'Pred']].to_csv(r'C:\\users\\Thomas\\workspace\\mm_ncaa2019\\src\\submission_scikit_NN_alldeltacols_alpha7_HLS900by1.csv', index=False)

