#!/usr/bin/env python
import psycopg2
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
from fastapi import FastAPI, HTTPException
app = FastAPI()

@app.get("/about")
def instructions():
  # Provide user with available parameter options
  with psycopg2.connect("host=A port=B dbname=C user=D password=E") as conn:
    with conn.cursor() as cur:
        try:
            cur.execute("""
            SELECT DISTINCT sector
            FROM afr_articles;
            """)
            sectors = cur.fetchall()
            cur.execute("""
            SELECT DISTINCT category
            FROM afr_articles;
            """)
            categories = cur.fetchall()
            cur.execute("""
            SELECT DISTINCT author
            FROM afr_articles;
            """)
            authors = cur.fetchall()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            cur.close()   
  return {
  "instructions":"Enter S and C parameters separated by a comma without a space. Enter A parameter with separate authors separated by a fullstop, and co-authors separated by a comma and space. Type parameter values with correct capitalization and internal spaces.",
  "sectors (default: all)":sectors,
  "categories (default: all)":categories,
  "authors (default: all)":authors
  }

def mse(a,b):
    # Calculate the mean squared error given two tensors
    return np.square(a.detach().cpu().squeeze().numpy()-b.detach().cpu().squeeze().numpy()).mean()

class GRUModel(nn.Module):
    # Define PyTorch class for a GRU followed by a linear layer
    def __init__(self, input_size, output_size, hidden_size, layer_size, device):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.GRU = nn.GRU(input_size, hidden_size, layer_size,batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
    
    def forward(self, x):
        hidden = torch.zeros(self.layer_size, x.size(0), self.hidden_size).requires_grad_().to(self.device)
        out, self.hidden = self.GRU(x,hidden.detach())
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)
        out = nn.Sigmoid()(out)
        return out

def postgres_query_tabulate(s='all',c='all',a='all'):
    # Query the database for relevant articles and extract the sentiment scores
    sectors = tuple(s.split(','))
    categories = tuple(c.split(','))
    authors = tuple(a.split('.'))
    with psycopg2.connect("host=ec2-52-63-12-173.ap-southeast-2.compute.amazonaws.com port=5432 dbname=misc user=hiring_test_readonly password=pretense_yarrow_armhole") as conn:
        with conn.cursor() as cur:
            try:
                if s=='all' and c=='all' and a=='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles;
                    """)
                elif s!='all' and c=='all' and a=='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (sector IN %s);
                    """,(sectors,))
                elif s=='all' and c!='all' and a=='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (category IN %s);
                    """,(categories,))
                elif s=='all' and c=='all' and a!='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (author IN %s);
                    """,(authors,))
                elif s!='all' and c!='all' and a=='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (sector IN %s) AND (category IN %s);
                    """,(sectors,categories))
                elif s=='all' and c!='all' and a!='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (category IN %s) AND (author IN %s);
                    """,(categories,authors))
                elif s!='all' and c=='all' and a!='all':
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (sector IN %s) AND (author IN %s);
                    """,(sectors,authors))
                else:
                    cur.execute("""
                    SELECT publish_datetime, sentiment_score
                    FROM afr_articles
                    WHERE (sector IN %s) AND (category IN %s) AND (author IN %s);
                    """,(sectors,categories,authors))
                rows = cur.fetchall()
                column_names = [desc[0] for desc in cur.description]
                df = pd.DataFrame(rows, columns=column_names)
                df['sentiment_score'] = df['sentiment_score'].apply(pd.to_numeric, downcast='float')
                return df
            except (Exception, psycopg2.DatabaseError) as error:
                print("Error: %s" % error)
                cur.close()   
                return None

def mean_sentiment_scores(s='all',c='all',a='all'):
    # Calculate the mean sentiment score for each article publication date
    df = postgres_query_tabulate(s,c,a)
    if len(df)==0:
        raise HTTPException(status_code=404, detail = "No articles match the requested parameters; try broadening the parameters.")
    else:
        df['publish_date'] = df['publish_datetime'].dt.date
        return df[['publish_date','sentiment_score']].groupby(by='publish_date').mean()

def data_partition(s='all',c='all',a='all',ws='28'):
    # Get the mean sentiment scores for each publication date and chunk into windows
    ws = int(ws)
    mss = mean_sentiment_scores(s,c,a)
    m = mss.shape[0]
    windowed_data = torch.tensor( [mss['sentiment_score'].iloc[i:i+ws].values for i in range(0,m-ws,int(ws/2))])
    shuffling = np.random.permutation(windowed_data.shape[0]) # shuffle to decouple sampled window from original ordering
    x_windowed = windowed_data[shuffling,:-1]
    y_windowed = windowed_data[shuffling,-1]

    # Partition the datasets, holding out 10% for validation and 10% for testing
    seam_1 = math.floor(0.8*x_windowed.shape[0])
    seam_2 = math.floor(0.9*x_windowed.shape[0])
    x_train = x_windowed[:seam_1,:].unsqueeze(0).float()
    x_val   = x_windowed[seam_1:seam_2,:].unsqueeze(0).float()
    x_test  = x_windowed[seam_2:,:].unsqueeze(0).float()
    y_train = y_windowed[:seam_1].unsqueeze(0).float()
    y_val   = y_windowed[seam_1:seam_2].unsqueeze(0).float()
    y_test  = y_windowed[seam_2:].unsqueeze(0).float()
    assert(x_train.shape[1]+x_val.shape[1]+x_test.shape[1]==x_windowed.shape[0])
    assert(x_train.shape[1]==y_train.shape[1] and x_val.shape[1]==y_val.shape[1] and x_test.shape[1]==y_test.shape[1])
    return x_train, x_val, x_test, y_train, y_val, y_test

def model_statistics(x1,x2,x3,y1,y2,y3):
    # Get performance statistics for train/dev/test datasets, formatted as tensors
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Instantiate a new model and load saved state from pre-trained model
    n = x1.shape[2]
    hidden_size = 7
    layer_size = 2
    model = GRUModel(n, 1, hidden_size, layer_size, device);
    model.load_state_dict(torch.load('./model.pt',map_location = device));
    model.to(device)
    model.eval(); # use model for inference
    mse_train = mse(model(x1.to(device)),y1).tolist()
    mse_val = mse(model(x2.to(device)),y2).tolist()
    mse_test = mse(model(x3.to(device)),y3).tolist()
    return mse_train,mse_val,mse_test

@app.get("/stats")
def model_stats():
    # Display MSE for the model on the three datasets
    # P.S. I realize the values change each time the function is called;
    # I should have fixed a random number seed for training and used the same seed to recreate the same data partitions

    x1,x2,x3,y1,y2,y3 = data_partition('all','all','all')
    mse_train,mse_val,mse_test = model_statistics(x1,x2,x3,y1,y2,y3)
    return {'about':'Performance on entire datasets (mean square error)','train':mse_train,'validation':mse_val,'test':mse_test}

@app.get("/predict")
def model_inference(sectors='all',categories='all',authors='all'):
    # Given the three filtering parameters, predict the mean sentiment score for the day following the dataset's last publication
    mss = mean_sentiment_scores(sectors,categories,authors)
    m = 27 # use the last m days' data to predict the next day's datum
    windowed_data = torch.tensor([mss['sentiment_score'].values]).unsqueeze(0)
    if windowed_data.shape[2]<m:
        windowed_data = torch.cat(( torch.zeros((1,1,m-windowed_data.shape[2])), windowed_data ),dim=2)
    dates = mss.index.astype(str).tolist()
    x = windowed_data.float()[:,:,-27:]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate a new model and load saved state from pre-trained model
    hidden_size = 7
    layer_size = 2
    model = GRUModel(m, 1, hidden_size, layer_size, device);
    model.load_state_dict(torch.load('./model.pt',map_location = device));
    model.to(device)
    model.eval(); # use model for inference
    return {'dates':dates, 'mean_sentiment_score':mss['sentiment_score'].values.tolist(), 'prediction':model(x.to(device)).detach().cpu().squeeze().numpy().tolist()}
