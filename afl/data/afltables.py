import numpy as np
import pandas as pd


def get_fixtures():
    """Retrieve all fixture data from afltables.com as a dataframe

    Returns
    -------
    df : pandas.DataFrame
        Each match, with matchid index and 19 columns relating to home_team, away_team and game metadata

    The steps are pretty much taken from https://github.com/jimmyday12/fitzRoy/blob/master/R/afltables_basic.R
    """

    # The delimiter seems to be padded whitespace. Unfortunately, sometimes it is a single space,
    # which is confusing because some fields have a single space, eg. 'South Melbourne'
    # A working rule is to delimit with either two or more spaces, or a '.' followed by one or more spaces
    # Respect the matchid used in the first column
    cols = ['matchid', 'date', 'round', 'home_team', 'home_score', 'away_team', 'away_score', 'venue']
    df = pd.read_csv('https://afltables.com/afl/stats/biglists/bg3.txt', skiprows=1, header=None, names=cols,
                     sep=r'\s\s+|\.\s+', engine='python', index_col='matchid')

    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
    df['season'] = df['date'].dt.year
    df['home_points'] = df['home_score'].str.split('.').str[-1].astype(int)
    df['away_points'] = df['away_score'].str.split('.').str[-1].astype(int)

    df['home_points_ratio'] = df['home_points'] / (df['home_points'] + df['away_points'])
    df['home_margin'] = df['home_points'] - df['away_points']

    df['winner'] = np.nan
    df.loc[df['home_margin'].gt(0), 'winner'] = 'home'
    df.loc[df['home_margin'].eq(0), 'winner'] = 'draw'
    df.loc[df['home_margin'].lt(0), 'winner'] = 'away'
    df['home_win_draw_loss'] = df['winner'].map({'home': 1, 'draw': 0.5, 'away': 0})

    finals = ['QF', 'EF', 'SF', 'PF', 'GF']
    df['round_type'] = np.nan
    df.loc[df['round'].str.contains(r'R\d'), 'round_type'] = 'regular'
    df.loc[df['round'].isin(finals), 'round_type'] = 'finals'

    df['round'] = df['round'].replace({'QF': 'QF/EF', 'EF': 'QF/EF'})
    rounds = [f'R{i}' for i in range(1, 25)] + ['QF/EF', 'SF', 'PF', 'GF']
    df['round_number'] = df['round'].map({label: i+1 for i, label in enumerate(rounds)})

    team_name_mapper = {
        'Kangaroos': 'North Melbourne',
        'NM': 'North Melbourne',
        'Western Bulldog': 'Footscray',
        'Western Bulldogs': 'Footscray',
        'WB': 'Footscray',
        'South Melbourne': 'Sydney',
        'Brisbane Bears': 'Brisbane Lions',
        'Lions': 'Brisbane Lions',
        'Brisbane': 'Brisbane Lions',
        'GW Sydney': 'GWS',
        'Greater Western Sydney': 'GWS',
        'GC': 'Gold Coast',
        'StK': 'St Kilda',
        'PA': 'Port Adelaide',
        'WCE': 'West Coast',
    }

    state_mapper = {
        **{t: 'VIC' for t in ['Fitzroy', 'Collingwood', 'Essendon',
                              'St Kilda', 'Melbourne', 'Carlton', 'Richmond',
                              'University', 'Hawthorn', 'North Melbourne', 'Footscray']},
        'Geelong': 'GEE',
        'Sydney': 'NSW',
        'GWS': 'NSW',
        'West Coast': 'WA',
        'Fremantle': 'WA',
        'Port Adelaide': 'SA',
        'Adelaide': 'SA',
        'Gold Coast': 'QLD',
        'Brisbane Lions': 'QLD',
    }

    df['home_team'] = df['home_team'].replace(team_name_mapper)
    df['away_team'] = df['away_team'].replace(team_name_mapper)

    df['home_state'] = df['home_team'].map(state_mapper)
    df['away_state'] = df['away_team'].map(state_mapper)

    df['is_interstate'] = (df['home_state'] != df['away_state'])
    return df


def get_players():
    pass
