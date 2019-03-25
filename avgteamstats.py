'''
@author: Thomas Hymel
Created to follow along a similar Kernel from Kaggle on the NCAA March Madness 2019 competition, in order 
to learn how to (and become more comfortable with the ability to) import data and modify it, creating derived rows
and other useful data for later use in  a machine learning training algorithm
'''

import numpy as np #importing NumPy as np
import pandas as pd #importing pandas as pd
import tourney_info as ti  #my own Python script that creates the tourney_results training data set
import adder_helper as add  #my own Python script that helps with adding additional rows from outside data/other .csvs
from tourney_info import tourney_results, submission_setup
from adder_helper import seeds_df

# Load Data from Regular Season Detailed Results. Contains all the data of 2003-2019
season_results_name = '../src/input_data/RegularSeasonDetailedResults.csv'
season_df = pd.read_csv(season_results_name, nrows = 10)    #limiting to only the first 10 rows for printing brevity
#print(season_df) #printing what the dataframe looks like inherently

#Creates a dictionary that holds the data types that pd thinks is in the season dataframe
season_dtypes = season_df.dtypes.to_dict()

season_df = pd.read_csv(season_results_name, dtype=season_dtypes)
#print(season_df.head()) #.head returns the first n rows. By default n=5. Here, returns first 5 rows.

'''
Here we collect the stats for each team
'''
#stat column names, useful for grabbing the correct columns later on
win_cols = ['Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'NumOT']
lose_cols = ['Season', 'LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'NumOT']

#Average stats per team for winning team games
#Uses the df[list_of_column_names] to grab only certain columns. Then .groupby lists the data by Season first, and then by WTeamID
#After that, the .agg is an alias for .aggregate. Can't figure out why 'mean', is in {}. Can also use 'min', 'max'
#.reset_index() simply resets the index of the returned dataframe
WTeam_ID_avg = season_df[win_cols].groupby(['Season', 'WTeamID']).agg({'mean'}).reset_index()

#number of wins per team - .agg looks into the WScore column, and returns the total count of how many there are
WTeam_counts = season_df.groupby(['Season', 'WTeamID']).agg({'WScore':'count'}).reset_index()

#Average stats per team for losing team games
LTeam_ID_avg = season_df[lose_cols].groupby(['Season', 'LTeamID']).agg({'mean'}).reset_index()

#number of losses per team
LTeam_counts = season_df.groupby(['Season', 'LTeamID']).agg({'LScore':'count'}).reset_index()

#rename columns:   - uses df.columns to access/modify the column names of the dataframe
col_names = ['Season', 'TeamID', 'PointsScored', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', \
             'AST', 'TO', 'Stl', 'Blk', 'PF', 'NumOT']
WTeam_ID_avg.columns= col_names
LTeam_ID_avg.columns= col_names

#rename columns for win/loss counts   - directly supplying a list of str to rename the columns
WTeam_counts.columns =['Season', 'TeamID', 'Wins']
LTeam_counts.columns =['Season', 'TeamID', 'Losses']

#now we want to merge the number of wins and losses with their respective average stats:

#indices should be the same due to us reseting index with .reset_index() above
#df.merge() takes a different dataframe and merges it in a database-style join. 
WTeam_ID_avg_merged = WTeam_ID_avg.merge(WTeam_counts, how='left', on=None)
LTeam_ID_avg_merged = LTeam_ID_avg.merge(LTeam_counts, how='left', on=None)

#Naming which columns we want to combine (ie, the actual data, not the identifier columns)
cols = ['PointsScored', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', \
        'AST', 'TO', 'Stl', 'Blk', 'PF', 'NumOT']
#Create a weighted average to correctly combine the wins/losses statistics .mul is a scalar multiplication for the entire dataframe
stats_wins = WTeam_ID_avg_merged[cols].mul(WTeam_ID_avg_merged['Wins'], axis=0)
stats_losses = LTeam_ID_avg_merged[cols].mul(LTeam_ID_avg_merged['Losses'], axis=0)
num_games = WTeam_ID_avg_merged['Wins'] + LTeam_ID_avg_merged['Losses']
#weighted average calculation
team_stats = (stats_wins + stats_losses).div(num_games, axis=0)

#team average final stats merged with the season, TeamID, wins, and losses
RecordTeam_counts = WTeam_counts.merge(LTeam_counts, how='left', on=None)
team_avgs = RecordTeam_counts.merge(team_stats, how='left', right_index=True, left_index=True)

team_avgs = team_avgs.fillna(0)

#add winning percentage to the team_avgs data frame
team_avgs['WinPercent'] = team_avgs['Wins'].div(team_avgs['Wins']+team_avgs['Losses'])

#add Field Goal Percentage to the team_avgs data frame
team_avgs['FGPercent'] = team_avgs['FGM'].div(team_avgs['FGA'])

#add Free Throw Percentage to the team_avgs data frame
team_avgs['FTPercent'] = team_avgs['FTM'].div(team_avgs['FTA'])

#Check the team_avgs dataframe for any missing values
dummy = add.missing_values(team_avgs)

#Define a function to help with adding information to tourney_results dataframe from the team_avgs
def avg_stats(ex, col_name,iter):
    #This function helps to add the columns of data to the main training set data that comes from the season averages for each team
    bool0 = (team_avgs['Season'] == ex['Season'])   #filter out the season of the row
    bool1 = (team_avgs['TeamID'] == ex['Team1'])    #boolean that chooses Team1 rows
    bool2 = (team_avgs['TeamID'] == ex['Team2'])    #boolean that chooses Team2 rows
    stat_Team1 = team_avgs[bool0 & bool1][col_name].iloc[0]     #returns scalar value of the specific Team1 stat
    stat_Team2 = team_avgs[bool0 & bool2][col_name].iloc[0]     #returns scalar value of the specific Team2 stat
    deltaStat = stat_Team1 - stat_Team2                         #delta stat 
    if iter == 1:
        return stat_Team1
    elif iter == 2:
        return stat_Team2
    elif iter == 3:
        return deltaStat


export_csv = team_avgs.to_csv(r'C:\\users\\Thomas\\workspace\\mm_ncaa2019\\src\\team_avgsTEST.csv', index=False, header=True)

'''
#CODE LIST TO ADD to final version of tourney_results df (training data columns)
'''
submission_setup['deltaSeed'] = submission_setup.apply(add.delta_seed,axis=1)
submission_setup['Team1_Seed'] = submission_setup.apply(add.team1_seed,axis=1)
submission_setup['Team2_Seed'] = submission_setup.apply(add.team2_seed,axis=1)
submission_setup['deltamOrd']  = submission_setup.apply(add.delta_seed,axis=1)
submission_setup['Team1_mOrd'] = submission_setup.apply(add.team1_mOrd,axis=1)
submission_setup['Team2_mOrd'] = submission_setup.apply(add.team2_mOrd,axis=1)

tourney_results['deltaSeed'] = tourney_results.apply(add.delta_seed,axis=1)
tourney_results['Team1_Seed'] = tourney_results.apply(add.team1_seed,axis=1)
tourney_results['Team2_Seed'] = tourney_results.apply(add.team2_seed,axis=1)
tourney_results['deltamOrd']  = tourney_results.apply(add.delta_mOrd,axis=1)
tourney_results['Team1_mOrd'] = tourney_results.apply(add.team1_mOrd,axis=1)
tourney_results['Team2_mOrd'] = tourney_results.apply(add.team2_mOrd,axis=1)


avg_cols_to_add = ['Wins', 'Losses', 'PointsScored', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', \
        'AST', 'TO', 'Stl', 'Blk', 'PF', 'NumOT', 'WinPercent', 'FGPercent', 'FTPercent']       #defines the columns in team_avgs to add to tourney_results

for col in avg_cols_to_add:
    for i in range(1,4):
        print("Doing", col, i)
        if i ==1:
            submission_setup['Team1_'+col] = submission_setup.apply(avg_stats, args=(col,i),axis=1)
        elif i==2:
            submission_setup['Team2_'+col] = submission_setup.apply(avg_stats, args=(col,i),axis=1)
        elif i==3:
            submission_setup['delta'+col] = submission_setup.apply(avg_stats, args=(col,i),axis=1)
 

#export_csv2 = tourney_results.to_csv(r'C:\\users\\Thomas\\workspace\\mm_ncaa2019\\src\\training_data.csv', index=False, header=True)
export_csv3 = submission_setup.to_csv(r'C:\\users\\Thomas\\workspace\\mm_ncaa2019\\src\\submission_setup_Stage2.csv', index=False, header=True)
print('Successfully exported .csv')