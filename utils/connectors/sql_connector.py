import psycopg2
import os 

"""
TO DO: 
Add INSERT JSON TO TABLE DATA 
USE COMPANY SHORT NAME MAPPING FROM JSON
"""
class SqlConnector: 

    def __init__(self, db_name) -> None:
        self.connection = self.connect_to_cloud_sql(db_name)
        self.cursor = self.connection.cursor
        
    def connect_to_cloud_sql(self, db_name):
        return psycopg2.connect(
            host = os.environ['PSG_HOST'], 
            user = os.environ['PSG_USER'], 
            password = os.environ['PSG_PSWD'], 
            database = db_name
        )

    