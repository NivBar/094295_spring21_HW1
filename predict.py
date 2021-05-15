#!/usr/bin/env python
# coding: utf-8

# In[19]:


import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import sklearn
import pickle



# In[2]:


def drop_cols(df):
    remove_cols = ['backdrop_path', 'id', 'imdb_id', 'original_title', 'poster_path', 'spoken_languages', 'status', 'tagline',
                   'video', 'vote_count', 'overview', 'Keywords', 'production_countries', 'title']
    return df.drop(columns=remove_cols)


# In[3]:


def get_list_from_json(df, col, key):
    genres_set = set()
    for idx,row in df.iterrows():
        gen_list = get_dict(row[col])
        gens = [g[key] for g in gen_list]
        genres_set.update(gens)
    return genres_set 


# In[4]:


def transform_bin_cols(df, col, key):
    json_list = list(get_list_from_json(df,col,key))
    row_list = []   
    for idx,row in df.iterrows():
        json_list_temp = get_dict(row[col])
        bin_lst = np.zeros(len(json_list))
        if json_list_temp:
            temp = [g[key] for g in json_list_temp]
            indexes = [json_list.index(g) for g in temp]
            for i in range(len(bin_lst)):
                if i in indexes:
                    bin_lst[i]=1
        row_list.append(bin_lst)
    bin_df = pd.DataFrame(row_list,columns=json_list).add_prefix('is_')
    res_df = pd.concat([df,bin_df],axis=1).drop(columns=[col])
    return res_df


# In[5]:


def get_dict(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


# In[6]:


def top_k(df, col, keys, k):
    hist_ = {}
    for idx,row in df.iterrows():
        dicts_list = get_dict(row[col])
        if not dicts_list:
            continue
        for dict_ in dicts_list:
            curr_val = dict_[keys[0]]
            if keys[0] != keys[-1]:
                curr_job = dict_[keys[1]]
                if curr_job != 'Director':
                    continue
            if curr_val in hist_:
                hist_[curr_val] += 1
            else:
                hist_[curr_val] = 0
    top_k_keys = sorted(hist_, key=hist_.get, reverse=True)[:k]
    return top_k_keys


# In[7]:


def feature_transtormation(df, top_5_production_companies=[], top_100_actors=[], top_20_directors=[]):
    extended_train = transform_bin_cols(df,'genres','name')
    extended_train['is_en_original_language'] = np.where(extended_train.original_language == 'en', 1, 0)
    extended_train['is_homepage_exists'] = np.where(extended_train.homepage.isnull(), 0, 1)
    extended_train.release_date = np.where(extended_train.release_date.isnull(), "2013-09-25", extended_train.release_date)
    extended_train[['release_year', 'release_month', 'release_day']] = extended_train['release_date'].str.split('-',expand=True).replace(np.nan, -1).astype(int)
    if not top_100_actors:
        top_5_production_companies = top_k(extended_train, 'production_companies', ['name'], k=5)
        top_100_actors = top_k(extended_train, 'cast', ['name'], k=100)
        top_20_directors = top_k(extended_train, 'crew', ['name', 'job'], k=20)

    extended_train['top_directors'] = extended_train['crew'].apply(lambda y: sum([1 if x in str(y) else 0 for x in top_20_directors]))
    extended_train['top_prod_companies'] = extended_train['production_companies'].apply(lambda y: sum([1 if x in str(y) else 0 for x in top_5_production_companies]))
    extended_train['top_actors'] = extended_train['cast'].apply(lambda y: sum([1 if x in str(y) else 0 for x in top_100_actors]))
    extended_train.drop(['homepage','original_language','release_date', 'production_companies', 
                    'cast', 'crew','release_day' ], axis=1, inplace=True)
    return extended_train, top_5_production_companies, top_100_actors, top_20_directors


# In[8]:


def handle_missing_data(df, vote_average=-1, popularity=-1, budget_mean=-1, runtime_mean=-1):
    if vote_average == -1:
        vote_average = df['vote_average'].dropna(how='any').mean()
        popularity = df['popularity'].dropna(how='any').mean()
        budget_mean = df.loc[df['budget'] > 0,'budget'].mean()
        runtime_mean = df['runtime'].dropna(how='any').mean()
    
    df.vote_average = np.where(df.vote_average.isnull(), vote_average, df.vote_average)
    df.popularity = np.where(df.popularity.isnull(), popularity, df.popularity)
    df.budget = np.where(df.budget <= 0.0, budget_mean, df.budget)
    df.runtime = np.where(df.runtime.isnull(), runtime_mean, df.runtime)
    df.belongs_to_collection = np.where(df.belongs_to_collection.isnull(), 0, 1)
    
    return df, vote_average, popularity, budget_mean, runtime_mean


# In[9]:


def rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error between two input vectors
    :param y_true: 1-d array, ground truth vector
    :param y_pred: 1-d array, prediction vector
    :return: float, RMSLE score between two input vectors
    """
    assert y_true.shape == y_pred.shape,         ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
    return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


# ### Test Phase

# In[14]:


budget_mean_ = 30047340.334763948
runtime_mean_ = 108.02744194972175
vote_average_ = 6.3990987535954
popularity_ = 10.016558581016266
top_5_production_names_ = ['Warner Bros. Pictures', 'Universal Pictures', 'Paramount', 'Columbia Pictures',
                           '20th Century Fox']

top_100_actors_ = ['Samuel L. Jackson', 'Frank Welker', 'Morgan Freeman', 'Bruce Willis', 'Steve Buscemi',
                   'Robert De Niro', 'Nicolas Cage', 'John Goodman', 'Liam Neeson', 'Willem Dafoe', 'Matt Damon',
                   'Brad Pitt', 'Dennis Quaid', 'Alec Baldwin', 'Tommy Lee Jones', 'Keith David', 'Richard Jenkins',
                   'J.K. Simmons', 'John Leguizamo', 'Robin Williams', 'Johnny Depp', 'Bill Murray', 'Stanley Tucci',
                   'Antonio Banderas', 'Sylvester Stallone', 'Woody Harrelson', 'Ed Harris', 'Tom Wilkinson',
                   'Christopher Walken', 'Robert Downey Jr.', 'Keanu Reeves', 'Stephen Root', 'Clint Eastwood',
                   'Dennis Hopper', 'Julianne Moore', 'Ben Affleck', 'Christopher Plummer', 'Tom Hanks',
                   'Susan Sarandon', 'John Turturro', 'Ethan Hawke', 'Bruce Greenwood', 'James Franco', 'Helen Mirren',
                   'Brian Cox', 'John Lithgow', 'John Cusack', 'Philip Seymour Hoffman', 'Allison Janney',
                   'Bruce McGill', 'Arnold Schwarzenegger', 'Mickie McGowan', 'Denzel Washington', 'John Hurt',
                   'Sherry Lynn', 'Joe Pantoliano', 'Bill Hader', 'James Earl Jones', 'Nicole Kidman', 'Anthony Mackie',
                   'William H. Macy', 'Bill Paxton', 'Meryl Streep', 'Paul Giamatti', 'Robert Duvall', 'Dustin Hoffman',
                   'Dan Aykroyd', 'Julia Roberts', 'Gene Hackman', 'Vince Vaughn', 'Ralph Fiennes', 'Harrison Ford',
                   'Matthew McConaughey', 'David Koechner', 'Danny Glover', 'Harry Dean Stanton', 'Jack Angel',
                   'Jeff Bridges', 'Sean Connery', 'Andy García', 'John Franchi', 'Brendan Gleeson', 'Ben Stiller',
                   'Cate Blanchett', 'Alfred Molina', 'Bob Bergen', 'Tom Cruise', 'Elizabeth Banks', 'Eddie Murphy',
                   'Kevin Spacey', "Vincent D'Onofrio", 'Scarlett Johansson', 'Harvey Keitel', 'Mark Strong',
                   'Michael Gambon', 'Jude Law', 'Liev Schreiber', 'Gary Oldman', 'Jonah Hill', 'Bradley Cooper']

top_20_directors_ = ['Clint Eastwood', 'Steven Spielberg', 'Ridley Scott', 'Robert Rodriguez', 'Alfred Hitchcock',
                     'Ron Howard', 'Woody Allen', 'Richard Donner', 'Steven Soderbergh', 'Sidney Lumet',
                     'Blake Edwards', 'Robert Zemeckis', 'Wes Craven', 'Oliver Stone', 'Renny Harlin', 'Rob Reiner',
                     'Quentin Tarantino', 'Joel Schumacher', 'Tim Burton', 'Garry Marshall']


# In[16]:

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")

# data = pd.read_csv('test.tsv', sep="\t")
test = drop_cols(data)
test, _, _, _ = feature_transtormation(test, top_5_production_names_, top_100_actors_, top_20_directors_)
final_test, _, _, _, _ = handle_missing_data(test, vote_average_, popularity_, budget_mean_, runtime_mean_)
X_test, Y_test = final_test.loc[:, final_test.columns != 'revenue'], final_test['revenue']


# In[20]:


rmsle_ = make_scorer(rmsle)

with open('best_model.pickle', 'rb') as model_:
    b = pickle.load(model_)


# In[ ]:

print('initiating...')
df_pred = pd.DataFrame(b.predict(X_test), columns=['score'])
df_pred['id'] = data['id']
df_pred.score = np.where(df_pred.score < 0.0, 0, df_pred.score)
df_pred = df_pred[['id','score']]
df_pred.to_csv("prediction.csv", index=False, header=False)
print('successful')

