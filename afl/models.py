from collections import defaultdict
from math import exp, log

import pandas as pd  # optional


class Elo:
    """Base class to generate elo ratings

    Includes the ability for some improvements over the original methodology:
        * k decay: use a higher update speed early in the season
        * crunch/carryover: shift every team's ratings closer to the mean between seasons
        * interstate and home_advantage
        * optimised initial ratings

    Hyperparameters can be fit with a grid search, eg. sklearn.model_selection.GridSearchCV
    Initial ratings can be fit with a logistic regression (equivalent to a static elo) eg. sklearn.linear_model.LogisticRegression

    By default assumes a logistic distribution of ratings
    """
    def __init__(self, k=30, home_advantage=20, interstate_advantage=5, width=400/log(10), carryover=0.75, k_decay=0.95,
                 initial_ratings=None, mean_rating=1500, target='home_win_draw_loss'):
        self.k = k
        self.home_advantage = home_advantage
        self.interstate_advantage = interstate_advantage
        self.width = width
        self.carryover = carryover
        self.k_decay = k_decay

        self.mean_rating = mean_rating
        self.initial_ratings = initial_ratings or {}
        self.target = target  # home_win_draw_loss, home_points_ratio, home_squashed_margin

    def iterate_fixtures(self, fixtures, as_dataframe=True):
        """
        Parameters
        ----------
        fixtures : list of dict or pd.DataFrame
            Must be ordered. Each record (row) must have (columns): home_team, away_team, round_number, is_interstate, <self.target>
            Prefer a list of records as it's much faster

        We use the python stdlib math.exp which seems faster in single computation than numpy's version and therefore speeds up parameter fitting

        Profile code with lprun:
        %load_ext line_profiler
        elo = Elo()
        %lprun -f elo.iterate_fixtures elo.iterate_fixtures(fxtrain, as_dataframe=True)
        """
        # new teams are given self.initial_ratings
        self.current_ratings_ = defaultdict(lambda: self.mean_rating, self.initial_ratings)

        if isinstance(fixtures, pd.DataFrame):
            # A list of records is faster and less prone to errors on update than a DataFrame
            fixtures = fixtures.reset_index().to_dict('records')

        for fx in fixtures:
            home_team = fx['home_team']
            away_team = fx['away_team']
            home_actual_result = fx[self.target]
            round_number = fx['round_number']
            is_interstate = fx['is_interstate']

            # home_expected_result = self.predict_result(home_team, away_team, is_interstate, round_number)
            # -------
            home_rating_pre = self.current_ratings_[home_team]
            away_rating_pre = self.current_ratings_[away_team]

            if round_number == 1:
                # Crunch the start of the season
                # Warning: this will make an in-place change the current ratings for the end of season
                # TODO: don't crunch the first round of training
                home_rating_pre = self.carryover*home_rating_pre + (1-self.carryover)*self.mean_rating
                away_rating_pre = self.carryover*away_rating_pre + (1-self.carryover)*self.mean_rating

            ratings_diff = home_rating_pre - away_rating_pre + self.home_advantage + self.interstate_advantage*is_interstate
            home_expected_result = 1.0 / (1 + exp(-ratings_diff/self.width))

            # self.update_ratings(home_actual_result, home_expected_result, round_number)
            # ------
            change_in_home_elo = self.k*self.k_decay**round_number*(home_actual_result - home_expected_result)

            home_rating_post = home_rating_pre + change_in_home_elo
            away_rating_post = away_rating_pre - change_in_home_elo

            # update ratings
            self.current_ratings_[home_team] = home_rating_post
            self.current_ratings_[away_team] = away_rating_post

            fx['home_rating_pre'] = home_rating_pre
            fx['away_rating_pre'] = away_rating_pre
            fx['home_expected_result'] = home_expected_result  # proba
            # fx['binary_expected_home_result'] = int(expected_home_result > 0.5)  # prob

        if as_dataframe:
            # return pd.DataFrame(fixtures, columns=['matchid', 'home_expected_result']).set_index('matchid')
            return pd.DataFrame(fixtures).set_index('matchid')

        return fixtures

    def fit(self, X):
        # the only thing we really need to store is the *latest* rating (the system is memoryless)
        # self.teams_ = ['myteam']
        # self.current_ratings_ = {'myteam': 1500}
        return X

    def predict_proba(self):
        return expected_home_result

    def predict(self):
        return int(expected_home_result > 0.5)
