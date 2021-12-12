import pandas as pd
from psaw import PushshiftAPI
import praw 
from datetime import datetime 

class RedditPsawConnector:

    def __init__(self, 
                game_names, 
                ticker, 
                save_path, 
                subreddit='Games', 
                limit=2000,
                start_date='2018-12',
                end_date='2021-12'
                ): 

        self.subreddit = subreddit
        self.api = PushshiftAPI()
        self.game_names = self.prepare_query(game_names)
        self.ticker = ticker 
        self.limit = limit 
        self.save_path = save_path
        self.empty_df = self.prepare_empty_df()
        self.date_pairs = self.prepare_date_pairs(start_date, end_date)
        # save empty csv 
        self.save_df_to_csv(df=self.empty_df, mode='w')

    @staticmethod
    def prepare_query(game_names): 
        return '|'.join(game_names)

    @staticmethod
    def prepare_empty_df():
        return pd.DataFrame(
            columns= [
                'author', 'body', 'created_utc', 
                'score', 'created'
            ]
        )

    @staticmethod
    def prepare_date_pairs(start_date, end_date): 
        dates = pd.date_range(start_date,end_date, 
              freq='MS').strftime("%Y-%m").tolist()

        return [list(map(lambda x: int(x), date.split('-'))) for date in dates]

    def save_df_to_csv(self, df, mode='a'):
        df.to_csv(self.save_path, mode=mode, header=False if mode == 'a' else True)

    def search_comments(self, query, after, before, limit):
        filter = ['author', 'date', 'title', 'body', 'score']

        comments = self.api.search_comments(
            subreddit = self.subreddit,
            filter=filter, q=query, limit=limit, 
            sort_type="score", sort="desc", 
            after=after, before=before
        )
        return pd.DataFrame([comment.d_ for comment in comments])
 
    def batch_comment_search(self): 
        for i in range(len(self.date_pairs) - 1):
            # ugly but simple way to get right dates 
            start_year = self.date_pairs[i][0]
            start_month = self.date_pairs[i][1]
            end_year = self.date_pairs[i + 1][0]
            end_month = self.date_pairs[i + 1][1]

            df = self.search_comments(
                query=self.game_names, 
                after=int(datetime(start_year, start_month, 10).timestamp()), 
                before=int(datetime(end_year, end_month, 10).timestamp()), 
                limit=2000
            )
            self.save_df_to_csv(df, mode='a')




