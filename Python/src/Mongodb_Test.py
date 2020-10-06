from pymongo import MongoClient
from pymongo import errors
from bson.objectid import ObjectId


def get_credentials():
    credentials = {}
    # Open file
    with open("../data/dbcredentials.txt", "r") as creds:
        lines = creds.readlines()
    
    # Read lines
    for line in lines:
        cred = line.split()
        label = cred[0]
        value = cred[1]
        credentials[label] = value
    
    # Put into dictionary
    return credentials

def connect_client(creds):
    username = creds["username"]
    password = creds["password"]
    dbname = creds["dbname"]
    uri = "mongodb+srv://{}:{}@cluster0.zrbnr.mongodb.net/{}?retryWrites=true&w=majority".format(username, 
        password, dbname)
    client = MongoClient(uri)
    db = client.twitter_app
    return db

if __name__ == '__main__':
    creds = get_credentials()
    db = connect_client(creds)
    collection = db.tweets

    suffix = '12345'
    document = {
        '_id': ObjectId('1312534567597731840' + suffix),
        "key": "With custom oid",
        "updated": "inserting into tweet database"
    }
    try:
        collection.insert_one(document)
    except errors.DuplicateKeyError:
        print("Already inserted")
    print(collection.find_one({'_id': ObjectId('1312534567597731840' + suffix)}))
