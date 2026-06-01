# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:36:03 2026

@author: duca
"""


"""
1. Za svakog klijenta koji je ušao u default, odrediti mesec pre ulaska u default u kome je imao najveći pad kreditnog skora u prethodna 3 meseca.
2. Pronaći klijente kod kojih je iskoriscenost kredita porastao više od 40%, a kreditna sposobnost opala za više od 100 poena u periodu od 3 meseca pre ulaska u default status.
3. Za svaku credit score grupu korisnika odrediti prosecan odnos izmedju odobrene sume kredita i ukupne sume izvrsenih uplata, kao i stopu defaulta.
4. Za svakog klijenta izračunati najduži uzastopni niz meseci bez propuštene uplate.
5. Pronaći profil korisnika (kombinacija starosne grupe, pola, zaposlenosti i opsega godišnjih prihoda) koji ima najmanji prosečan DTI i najveći procenat urednih uplata tokom prvih 12 meseci otplate.

"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
#%% Grafik korelacije

def plot_correlation_for_col(df, col_name):
    plt.figure(figsize=(12,6)) 
    correlation_matrix = df.corr() 
    sorted_col_corr = correlation_matrix[col_name].sort_values(ascending=True) 
    sorted_col_corr = sorted_col_corr.drop(col_name)
    sb.barplot(x=sorted_col_corr.index, y=sorted_col_corr.values, palette='RdBu')
    plt.xticks(rotation=90);
    plt.tight_layout()
    plt.show()

#%% POSTAVLJANJE PUTANJA
base_path = Path(__file__).parent / 'podaci/'

path_credit = base_path / 'credit score/origination_data-2.csv'
path_credit_tracking = base_path / 'credit score/monthly_performance-2.csv'

print(path_credit ) 

#%% UCITAVANJE PODATAKA

df_credit_score = pd.read_csv(path_credit)

df_performance = pd.read_csv(path_credit_tracking)

#%% FOO ZA UPOZNAVANJE SA PODACIMA

def initial_evaluation(df):
    
    print ('INFORMACIJE O DATASETU:')
    print ('-----------------------------------------------------')
    
    print ('Oblik:', df.shape)
    
    print ('Kolone i tipovi:')
    
    print (df.dtypes)
    
    print ('Broj NA vrednosti')

    print (np.count_nonzero(df.isna()))    
    '''
    print ('Prvih i poslednjih 5 redova:')
           
    print (df.head(5))
    
    print (df.tail(5))
    
    print('Sample data:')
    
    print (df.sample(n=5))
    '''
#%%

initial_evaluation(df_credit_score) 

#%%
initial_evaluation(df_performance) 

#%%

print("MONTHLY\n",df_performance.dtypes)
#%%
print("ON ORIGIN\n",df_credit_score.dtypes)

#%%
print (np.unique(df_performance['dpd_current'])) 


#%%


def backfill_dpd_progression(df):


    df = df.sort_values(
        ["customer_id", "months_on_book"]
    ).reset_index(drop=True)


    for cust_id, idx in df.groupby("customer_id").groups.items():

        customer_rows = df.loc[idx]

        ninety_rows = customer_rows[
            customer_rows["dpd_current"] >= 90
        ]

        if ninety_rows.empty:
            continue

        first_default_idx = ninety_rows.index[0]

        pos = customer_rows.index.get_loc(first_default_idx)

        if pos >= 1:
            idx_60 = customer_rows.index[pos - 1]

            if df.loc[idx_60, "dpd_current"] == 0:
                df.loc[idx_60, "dpd_current"] = 60

        if pos >= 2:
            idx_30 = customer_rows.index[pos - 2]

            if df.loc[idx_30, "dpd_current"] == 0:
                df.loc[idx_30, "dpd_current"] = 30
                
    return df

#%%

df_performance = backfill_dpd_progression(df_performance)


print (np.unique(df_performance['dpd_current'])) 


#%%
print (np.unique(df_credit_score['employment_status']))
    
    
#%%plot_correlation_for_col

df_kljucna_polja = df_performance.drop(['customer_id', 'observation_month'], axis = 1)

plot_correlation_for_col(df_kljucna_polja, 'has_defaulted')

#%%


def plot_correlation_heatmap(
    df,
    columns=None,
    method="pearson",
    figsize=(14, 10),
    cmap="coolwarm",
    annot=False
):

    if columns is None:
        corr_df = df.select_dtypes(include=["number"])
    else:
        corr_df = df[columns]

    corr_matrix = corr_df.corr(method=method)

    plt.figure(figsize=figsize)

    sb.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"{method.capitalize()} Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    return corr_matrix

#%%

plot_correlation_heatmap(df_kljucna_polja)

#%% SKIP CELL, MONGODB WORK START
#%% Saving records

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["banking_db"]

db.origination_data.insert_many(
    df_credit_score.to_dict("records")
)

db.monthly_performance.insert_many(
    df_performance.to_dict("records")
)



#%% example

pipeline = [
    {
        "$match": {
            "months_on_book": {"$lte": 12}
        }
    },
    {
        "$group": {
            "_id": "$customer_id",
            "avg_score": {
                "$avg": "$credit_score"
            }
        }
    }
]

stats = db.command(
    "aggregate",
    "monthly_performance",
    pipeline=pipeline,
    explain=True,
    cursor={}
)
print(stats)




#%%

print (np.count_nonzero(df_credit_score.loc['loan_term' == 36]))
