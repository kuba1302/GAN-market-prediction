import pandas as pd
from psaw import PushshiftAPI
import praw 

class RedditPsawConnector:

    def __init__(self, subreddit='Games'): 
        self.subreddit = subreddit
        self.api = PushshiftAPI()

    def search_comments(self, name, limit=1000):
        # returned columns 
        filter = ['author', 'date', 'title', 'body', 'score']

        #cConnect to api, and get data
        comments = self.api.search_comments(
            subreddit = self.subreddit,
            filter=filter, q=name, limit=limit
        )
        # Prepare dataframe
        return pd.DataFrame([comment.d_ for comment in comments])
 


class RedditPrawConnector: 

    def __init__(self, subreddit='Games') -> None:
        self.subreddit = subreddit
        self.client = praw.Reddit()