'''
@author: Thomas Hymel
Created to help affix the seeds of Team1 and Team2, and the delta of them, to the correct rows, given a dataframe containing columns of 
at least Team1,Team2,Season
Expanded to help affix all kinds of data to the tourney results (training set) dataframe
'''


import numpy as np #importing NumPy as np
import pandas as pd #importing pandas as pd

#Load seeds data in from proper seeds file
seeds_name = '../src/input_data/NCAATourneySeeds.csv'
seeds_df = pd.read_csv(seeds_name) 
#Since we only care about the data from 2003 and onwards, we get rid of the rest of the data, and reset the index, dropping the old index
seeds_df= seeds_df.loc[seeds_df['Season'] >=2003].reset_index(drop=True)
#Change the format of the seeds from the original file (W01a) to a usable integer, grabbing the str values of indices 1,2 only
seeds_df['Seed'] = pd.to_numeric(seeds_df['Seed'].str[1:3], downcast='integer', errors='coerce')

    #These are the functions that deal with adding info about teams' seeds
def team1_seed(ex):
    #This function takes in a row of data and returns the Team1 Seed
    bool0 = (seeds_df['Season'] == ex['Season'])   #creates a Boolean series that returns true where the seasons are the same
    bool1 = (seeds_df['TeamID'] == ex['Team1'])    #creates a Boolean series that returns true where the TeamID value is same as Team1 value
    team1_seed = seeds_df[bool0 & bool1]['Seed'].iloc[0]    #a scalar value that corresponds to the Seed value of the one row that returns true
    return team1_seed                               #returns that scalar value for Team1_seed
    
def team2_seed(ex):
    #This function takes in a row of data and returns the Team2 Seed to be added
    bool0 = (seeds_df['Season'] == ex['Season'])   #creates a Boolean series that returns true where the seasons are the same
    bool2 = (seeds_df['TeamID'] == ex['Team2'])    #creates a Boolean series that returns true where the TeamID value is same as Team1 value
    team2_seed = seeds_df[bool0 & bool2]['Seed'].iloc[0]    #a scalar value that corresponds to the Seed value of the one row that returns true
    return team2_seed                               #returns that scalar value for Team2_seed

def delta_seed(ex):
    #This function takes in a row of data and returns the delta between the two team seeds Team1Seed - Team2Seed
    bool0 = (seeds_df['Season'] == ex['Season'])   #creates a Boolean series that returns true where the seasons are the same
    bool1 = (seeds_df['TeamID'] == ex['Team1'])    #creates a Boolean series that returns true where the TeamID value is same as Team1 value
    bool2 = (seeds_df['TeamID'] == ex['Team2'])    #creates a Boolean series that returns true where the TeamID value is same as Team1 value
    delta_seed =  seeds_df[bool0 & bool1]['Seed'].iloc[0] -  seeds_df[bool0 & bool2]['Seed'].iloc[0]  #delta of the Seed value
    return delta_seed                           #returns the delta seed

#This section deals with importing and creating a helper function for the Massey Ordinals data
mOrd_name = '../src/input_data/MasseyOrdinals_thru_2019_day_128.csv'   #name of file to load for Massey Ordinals
mOrd = pd.read_csv(mOrd_name)                                           #reads into a dataframe
mOrd = mOrd[(mOrd['Season'] >= 2003) & (mOrd['RankingDayNum']==128)]    #reassigns only the rows that are 2003+ and only on day 128 (last day of ranking is most relevant)
    
    #These are the functions that deal with adding Mass Ordinals to the main data set
def team1_mOrd(ex):
    #This function adds Team 1's Mass Ord
    bool0 = (mOrd['Season'] == ex['Season'])    #boolean that filters the current ex's Season
    bool1 = (mOrd['TeamID'] == ex['Team1'])     #boolean that filters the current ex's Team1
    team1mOrd = mOrd[bool0 & bool1]['OrdinalRank'].mean() #calculates the mean Mass Ord for given ex Team1
    return team1mOrd
    
def team2_mOrd(ex):
    #This function adds Team 2's Mass Ord
    bool0 = (mOrd['Season'] == ex['Season'])    #boolean that filters the current ex's Season
    bool2 = (mOrd['TeamID'] == ex['Team2'])     #boolean that filters the current ex's Team2
    team2mOrd = mOrd[bool0 & bool2]['OrdinalRank'].mean() #calculates the mean Mass Ord for given ex Team2
    return team2mOrd

def delta_mOrd(ex):
    #This function adds the delta of the Mass Ord, Team 1 - Team 2
    bool0 = (mOrd['Season'] == ex['Season'])    #boolean that filters the current ex's Season
    bool1 = (mOrd['TeamID'] == ex['Team1'])     #boolean that filters the current ex's Team1
    bool2 = (mOrd['TeamID'] == ex['Team2'])     #boolean that filters the current ex's Team2
    team1mOrd = mOrd[bool0 & bool1]['OrdinalRank'].mean() #calculates the mean Mass Ord for given ex Team1
    team2mOrd = mOrd[bool0 & bool2]['OrdinalRank'].mean() #calculates the mean Mass Ord for given ex Team2
    deltamOrd = team1mOrd - team2mOrd   #calculates delta and returns that 
    return deltamOrd
    
def missing_values(df):
    if df.isnull().values.any():
        print('You have ' + str(df.isnull().sum().sum()) + ' missing values in this dataframe.')
        print(df.head())
        return df
    else:
        print('No missing values in this dataframe.')
        print(df.head())
        return df

