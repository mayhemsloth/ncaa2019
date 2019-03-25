'''
@author: Thomas
Created to simulate different tournaments to help fill out a single bracket based on the prediction table created
'''

import numpy as np #importing NumPy as np
import pandas as pd #importing pandas as pd

#Load prediction file in from proper files
LR_pred_name = '../src/submission_scikit_LR_alldeltacols_Ceqp026.csv'
NN_pred_name = '../src/submission_scikit_NN_alldeltacols_alpha7_HLS900by1.csv'
pred = pd.read_csv(LR_pred_name) 
#Load seeds file to assign seeds to TeamID
seeds_name = '../src/input_data/NCAATourneySeeds.csv'
seeds_original = pd.read_csv(seeds_name) 
seeds_df = seeds_original.copy()
#Load team name file to add
teams_name = '../src/input_data/Teams.csv'
teams_df = pd.read_csv(teams_name) 
#Load TourneySlots dataframe which will be the most
slots_name = '../src/input_data/NCAATourneySlots.csv'
slots_original = pd.read_csv(slots_name)
slots_original = slots_original.loc[slots_original['Season'] ==2019].reset_index(drop=True)  #we only care about the 2019 Season because that's what the file contains
slots = slots_original.copy()
#Load bracket template slots made by me for a nice visual representation of the bracket filled out
bracket_name = '../src/input_data/bracket_template_slots.csv'
bracket = pd.read_csv(bracket_name)
bracket = bracket.fillna(' ')

#create new columns that populate correctly with the strings of the numbers needed: Season, Team1, Team2
pred['Season'], pred['Team1'], pred['Team2'] = pred['ID'].str.split('_').str
#cast them to integers to be used later
pred[['Season', 'Team1', 'Team2']] = pred[['Season', 'Team1', 'Team2']].apply(pd.to_numeric)


def decide_game(ex):
    #function defined to randomly choose the winner of all games based on the prediction probability
    prob_Team1_wins = ex['Pred']
    if (np.random.random_sample() < prob_Team1_wins):
        return 1
    else:
        return 0

def add_1seed(ex):
    #function that adds the string Seeds of each team playing in a prediction game
    bool0 = (seeds_df['Season'] == ex['Season'])
    bool1 = (seeds_df['TeamID'] == ex['Team1'])
    team1_seed = seeds_df[bool0 & bool1]['Seed'].iloc[0] 
    return team1_seed
def add_2seed(ex):
    #function that adds the string Seeds of each team playing in a prediction game
    bool0 = (seeds_df['Season'] == ex['Season'])
    bool2 = (seeds_df['TeamID'] == ex['Team2'])
    team2_seed = seeds_df[bool0 & bool2]['Seed'].iloc[0] 
    return team2_seed
def add_1name(ex):
    bool1 = (teams_df['TeamID'] == ex['Team1'])
    team1_name = teams_df[bool1]['TeamName'].iloc[0]
    return team1_name
def add_2name(ex):
    bool2 = (teams_df['TeamID'] == ex['Team2'])
    team2_name = teams_df[bool2]['TeamName'].iloc[0]
    return team2_name
def add_name(ex):
    bool0 = (teams_df['TeamID'] == ex['TeamID'])
    team_name = teams_df[bool0]['TeamName'].iloc[0]
    return team_name
def update_seed(ex, bracket):
    #function that takes in the rows of the slots dataframe [Slots, StrongSeed, WeakSeed] and updates the seeds_df of the TeamID seed with the new slot seed
    strong_seed = ex['StrongSeed']    #don't need .iloc[0] for strings.
    print('strong_seed', strong_seed)
    weak_seed = ex['WeakSeed']
    print('weak_seed', weak_seed)
    season = ex['Season']
    bool0 = (seeds_df['Season'] == season)
    bools = (seeds_df['Seed'] == strong_seed)
    boolw = (seeds_df['Seed'] == weak_seed)
    bracket.replace(strong_seed, seeds_df[bool0 & bools]['TeamName'].iloc[0],inplace=True)
    bracket.replace(weak_seed, seeds_df[bool0 & boolw]['TeamName'].iloc[0] ,inplace=True)
    winner_ID = find_winner(strong_seed, weak_seed, season)
    winner_seedslot = ex['Slot']
    seeds_df.loc[ (seeds_df['Season'] == season) & (seeds_df['TeamID'] == winner_ID), 'Seed'] = winner_seedslot

def find_winner(strong, weak, season):
    #function that takes in two seeds and the season and returns the winner's TeamID
    bool0 = (seeds_df['Season'] == season)   #Boolean filters only the data that has
    bools = (seeds_df['Seed'] == strong)
    boolw = (seeds_df['Seed'] == weak)
    strong_TeamID = seeds_df[bool0 & bools]['TeamID'].iloc[0]
    print('Strong_TeamID', strong_TeamID)
    weak_TeamID = seeds_df[bool0 & boolw]['TeamID'].iloc[0]
    print('Weak_TeamID', weak_TeamID)
    ID_str = ''
    if strong_TeamID < weak_TeamID:         #reconstruct the ID to access the correct matchup in the LR/NN_pred
        ID_str = ID_str + str(season)+'_'+str(strong_TeamID)+'_'+str(weak_TeamID)
    else:
        ID_str = ID_str+ str(season)+'_'+str(weak_TeamID)+'_'+str(strong_TeamID)
    print('ID_str', ID_str)
    result = pred.loc[pred['ID']==ID_str]['Result'].iloc[0]
    print('result', result)
    if result == 1:
        print(pred[pred['ID']==ID_str]['Team1'].iloc[0], teams_df[teams_df['TeamID']==pred[pred['ID']==ID_str]['Team1'].iloc[0]]['TeamName'].values)
        return pred[pred['ID']==ID_str]['Team1'].iloc[0]
    else:
        print(pred[pred['ID']==ID_str]['Team2'].iloc[0], teams_df[teams_df['TeamID']==pred[pred['ID']==ID_str]['Team2'].iloc[0]]['TeamName'].values)
        return pred[pred['ID']==ID_str]['Team2'].iloc[0]

pred['Result'] = pred.apply(decide_game, axis=1) #make a prediction for every possible game that is played in the submission file for 2019

seeds_df['TeamName'] = seeds_df.apply(add_name, axis=1) #add the Team Name next to the TeamID in Seeds file

slots.apply(update_seed, args=(bracket,), axis=1)    #goes through the slots df, which informs what games to play and what team gets their seed updated based on winning

bracket.replace('R6CH', seeds_df[seeds_df['Seed'] =='R6CH']['TeamName'].iloc[0] ,inplace=True) #replace final Championship winner with TeamName
print(bracket.to_string())


    