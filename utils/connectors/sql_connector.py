import psycopg2
import os 

def connect_to_cloud_sql(db_name):
    return psycopg2.connect(
        host = os.environ['PSG_HOST'], 
        user = os.environ['PSG_USER'], 
        password = os.environ['PSG_PSWD'], 
        database = db_name
    )