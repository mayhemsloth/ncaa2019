'''
@author: Thomas Hymel
Created to follow along and produce my own version of the tourney information used to dictate the results (y=0,1) of the previous 
NCAA tournaments. 
REMEMBER: 
The submission file is yyyy_T1ID_T2ID, probability of T1 winning, where T1ID is ALWAYS less than T2ID
Therefore if WTeam_ID < LTeam_ID, that row gets a 1 for results
          if WTeam_ID > LTeam_ID, that row gets a 0 for results
'''

import numpy as np #importing NumPy as np
import pandas as pd #importing pandas as pd

#Load data in from the compact tourney results 
tourney_results_name = '../src/input_data/NCAATourneyCompactResults.csv'
tourney_df = pd.read_csv(tourney_results_name) 
#Since we only care about the data from 2003 and onwards, we get rid of the rest of the data, and reset the index, dropping the old index
tourney_df = tourney_df.loc[tourney_df['Season'] >=2003].reset_index(drop=True)

#Create and populate the tourney results dataframe in the correct way
tourney_results = pd.DataFrame()
tourney_results['Result'] = np.zeros(len(tourney_df.index),dtype=int)   #initialize all the results to 0
tourney_results['Season'] = tourney_df['Season'].values   #put the Season column into the results df
for i in range(len(tourney_results.index)):                 #runs through the rows and checks if WTeamID < LTeamID, if true, change result to 1
    if tourney_df.loc[i,'WTeamID'] < tourney_df.loc[i,'LTeamID']:
        tourney_results.loc[i,'Result'] = 1

#Add the correct Team1 and Team2 designation based on the results
tourney_results['Team1'] = tourney_results['Result'].values * tourney_df['WTeamID'].values \
        + (1-tourney_results['Result'].values) * tourney_df['LTeamID'].values
tourney_results['Team2'] = (1-tourney_results['Result'].values) * tourney_df['WTeamID'].values \
        + (tourney_results['Result'].values) * tourney_df['LTeamID'].values

'''We now have a tourney_results df that has Result (0 or 1), Season, Team1, Team2 columns. 
We should use this tourney results df as the base to add on additional information retrieved from other sources/dfs
when we make the predictions, in much the same way, we'll have a tourney info df that contains the Seasons, Team1, and Team2, which 
can be used to add on additional information about Season 2019 for each of those teams
'''

#Load data in from the SampleSubmissionStage2 file to help give us our 2019 data of Season (2019), Team1, Team2.
submission_setup_name = '../src/input_data/SampleSubmissionStage2.csv'
submission_setup = pd.read_csv(submission_setup_name) 

#create new columns that populate correctly with the strings of the numbers needed: Season, Team1, Team2
submission_setup['Season'], submission_setup['Team1'], submission_setup['Team2'] = submission_setup['ID'].str.split('_').str
#cast them to integers
submission_setup[['Season', 'Team1', 'Team2']] = submission_setup[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)
print(submission_setup.head())


