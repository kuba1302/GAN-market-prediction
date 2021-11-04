import pandas as pd
from psaw import PushshiftAPI


class RedditApiConnector:

    def __init__(self, subreddit='movies'):
        self.subreddit = subreddit
        self.api = PushshiftAPI()

    def search_comments(self, name, limit=1000):
        self.name = name
        # Hardcode filters to keep same columns in dataframes
        filter = ['author', 'date', 'title', 'body', 'score']

        # Connect to api, and get data
        comments = self.api.search_comments(
            subreddit = self.subreddit,
            filter=filter, q=self.name, limit=limit
        )
        # Prepare dataframe
        df = pd.DataFrame([comment.d_ for comment in comments])

        return df