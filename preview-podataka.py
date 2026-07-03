# -*- coding: utf-8 -*-
"""
Created on Tue May 26 20:36:03 2026

@author: duca
"""


"""
1. Za svakog klijenta koji je ušao u default, odrediti mesec pre ulaska u default u kom je imao najveći pad kreditnog rejtinga u prethodna 3 meseca.
2. Pronaći klijente kod kojih je iskorišćenost kreditne kartice porastao više od 40%, a kreditna sposobnost opala za više od 100 poena u periodu od 3 meseca pre likvidacije.
3. Za korisnike sa lošim kreditnim rejtingom u trenutku odobrenja kredita (<579) odrediti odnos ukupne sume izvršenih uplata i predviđene vrednosti uplata u prvih 12 meseci, kao i stopu defaulta.
4. Za korisnike sa izuzetnim kreditnim rejtingom u trenutku odobrenja kredita (>800) pronaći broj korisnika čiji je rejting u nekom trenutku opao u nižu kategoriju (<=799).
5. Za svakog klijenta izračunati najduži uzastopni niz meseci bez propuštene uplate.
6. Pronaći profil korisnika (starosnu grupu, rod, status zaposlenosti i opseg godišnjih prihoda) koji ima najmanji prosečan DTI i najveći procenat urednih uplata tokom prvih 12 meseci otplate.
"""



"""
Lista ideja za indekse i rekonstruisanje:

- potencijalno smanjenje okvira podataka je i uklanjanje korisnika posle defaultovanja? ili sortiranje po has _defaulted i onda pretraga samo dok je 0 (ili do prvog =1)
1. index nad customer_id - svakako je foreign key, mada selektivnost je ~17% tkd moze biti i da nije toliko idealan al kao za joinove (doduse nardeni koraci mogu ga prikazati kao suvisnog) 
2. extended reference na podatke iz origination kolekcije, gender, income itd.
3. dodavanje calculated polja za 3m razlike i slicne stvari! 

"""
#%% basic imports
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
#%% Target col correlation graph

def plot_correlation_for_col(df, col_name):
    plt.figure(figsize=(12,6)) 
    correlation_matrix = df.corr() 
    sorted_col_corr = correlation_matrix[col_name].sort_values(ascending=True) 
    sorted_col_corr = sorted_col_corr.drop(col_name)
    sb.barplot(x=sorted_col_corr.index, y=sorted_col_corr.values, palette='RdBu')
    plt.xticks(rotation=90);
    plt.tight_layout()
    plt.show()
    
#%% Correlation matrix


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

#%% Basic data information

def initial_evaluation(df):
    
    print ('INFORMACIJE O DATASETU:')
    print ('-----------------------------------------------------')
    
    print ('Oblik:', df.shape)
    
    print ('Kolone i tipovi:')
    
    print (df.dtypes)
    
    print ('Broj NA vrednosti')

    print (np.count_nonzero(df.isna()))    

    print ('Prvih i poslednjih 5 redova:')
           
    print (df.head(5))
    
    print (df.tail(5))
    
    print('Sample data:')
    
    print (df.sample(n=5))



#%% fixing basic dpd inconsistencies


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
                df.loc[idx_60, "payment_made"] = 0
                df.loc[idx_60, "payment_amount"] = 0                

        if pos >= 2:
            idx_30 = customer_rows.index[pos - 2]

            if df.loc[idx_30, "dpd_current"] == 0:
                df.loc[idx_30, "dpd_current"] = 30
                df.loc[idx_30, "payment_made"] = 0
                df.loc[idx_30, "payment_amount"] = 0

        later_ninety_idxs = ninety_rows.index[1:]
        df.loc[later_ninety_idxs, "dpd_current"] = 120


    return df

#%% additional dpd tweaks to spice up 5th query
'''
NAMERNO ZAMUCIVANJE INACE DOBRIH ISPLATA ZA NEKE OD KORISNIKA

za ~30% klijenata raznih kategorija, ubaciti instancu kasnjenja sa placanjem
a zatim dodatne isplate naredni mesec u vidu duple uplate rate

u trenutku kada zakasne sa placanjem, azurirati:

credit_score oboriti za 5-10 (rand, u zavisnosti sa kasnjenjem)
payment_amount 0
payment_made 0
dpd_current +30

za sledeci mesec azurirati:
payment made = 1
payment amount = 2 * scheduled_emi
credit_score podici za ~ 7 poena

LOGIKA: 
    razdvojiti 2. dataset po korisnicima, samplovati ~30% grupa
    za svakog korisnika, izvuci jedan nasumican red i za njega odrediti da nisu platili ratu
    za naredni mesec, realizvoati nadoknadu propusta
    
'''

def late_payments(df):
    
    # razdvojiti 2. dataset po korisnicima, samplovati ~30% grupa
    
    unique_users = df_credit_score['customer_id'].unique()
    sample_users = np.random.choice(unique_users, int(len(unique_users)/3))
    
    #za svakog korisnika, izvuci jedan nasumican red i za njega odrediti da nisu platili ratu
    # specificno izostavljamo poslednja 4 meseca zbog ranijeg korigovanja za korisnike koji su likvidirani
    
    for user in sample_users:
        rows = df_performance[df_performance['customer_id'] == user][:-4]

        if (rows.empty):
            continue

        month_pick = np.random.choice(rows.index)
        
        # unosenje kasnjenja
        credit_reduction = np.random.randint(5, 10)
        
        df.at[month_pick, 'payment_amount'] = 0
        df.at[month_pick, 'payment_made'] = 0
        df.at[month_pick, 'dpd_current'] = df.at[month_pick, 'dpd_current'] + 30
        df.at[month_pick, 'credit_score'] = df.at[month_pick, 'credit_score'] - credit_reduction
        
        # peglanje vrednosti sledeci mesec
        
        df.at[month_pick + 1, 'payment_amount'] = df.at[month_pick + 1, 'scheduled_emi'] * 2 
        df.at[month_pick + 1, 'payment_made'] = 1
        df.at[month_pick + 1, 'dpd_current'] = 0
        df.at[month_pick + 1, 'credit_score'] = df.at[month_pick, 'credit_score'] + credit_reduction*1.2
        
    return df


#%% POSTAVLJANJE PUTANJA
base_path = Path(__file__).parent / 'podaci/'

path_credit = base_path / 'credit score/origination_data-2.csv'
path_credit_tracking = base_path / 'credit score/monthly_performance-2.csv'

#%% UCITAVANJE PODATAKA

df_credit_score = pd.read_csv(path_credit)

df_performance = pd.read_csv(path_credit_tracking)

#%% UPOZNAVANJE SA PODACIMA

initial_evaluation(df_credit_score) 

initial_evaluation(df_performance) 

#%% OSNOVNO SREDJIVANJE DPD

print (np.unique(df_performance['dpd_current'])) 

df_performance = backfill_dpd_progression(df_performance)


print (np.unique(df_performance['dpd_current'])) 

#%% DODATNO SREDJIVANJE DPD 
'''
rows = df_performance[df_performance['customer_id'] == 'CUST100007']

print(rows[['dpd_current', 'payment_made', 'payment_amount']])

df_performance = late_payments(df_performance)

print(rows[['dpd_current', 'payment_made', 'payment_amount']])
'''

#%% UKLANJANJE 3M KOLONA
    
print (df_performance.columns)    

df_performance.drop(['credit_score_delta_3m', 'revolving_util_delta_3m'], axis = 1, inplace = True)
#%%KORELACIJE

df_kljucna_polja = df_performance.drop(['customer_id', 'observation_month'], axis = 1)

plot_correlation_for_col(df_kljucna_polja, 'has_defaulted')

plot_correlation_heatmap(df_kljucna_polja)

#%% PROVERA SELEKTIVNOSTI KATEGORICKIH KOLONA - Mainly zarad informacija za indeksiranja

print (np.unique(df_credit_score['employment_status']))
#tba

#%% SKIP CELL, MONGODB WORK START
#%% Filling in collections

from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://localhost:27017/")
db = client["banking_db"]

#%% Making collections
db.origination_data.insert_many(
    df_credit_score.to_dict("records")
)

db.monthly_performance.insert_many(
    df_performance.to_dict("records")
)

print ("Insert complete!")

#%%
collection_og = db['origination_data']
collection_performance = db['monthly_performance']
#%% celija za proveravanje imena kolona datoteka

print(df_performance.columns)

#%% OSNOVNI UPITI NAD NEOPTIMIZOVANOM SHEMOM
#%% 1. Za svakog klijenta koji je ušao u default, odrediti mesec pre ulaska u 
# default u kom je imao najveći pad kreditnog rejtinga u odnosu na 3 meseca ranije.

# ovaj upit izvrsava se iskljucivo nad performance kolekcijom

pipeline_q1 = [
    {
        "$match" : {"has_defaulted" : 1}
    },
    {
        "$setWindowFields" : {
            "partitionBy" : "$customer_id",
            "sortBy" : {"observation_month" : 1},
            "output" : {"3m_frame" : {
                "$shift" : {
                    "output": "$credit_score",
                    "by": -3
                    }
                }
            }
        }
    },

    {
        "$addFields": {
            "drop_3m": {
                "$cond": [
                    {"$ne": ["$3m_frame", None]},
                    {"$subtract": ["$3m_frame", "$credit_score"]},
                    None
                    ]
                }
            }
    },
    {
        "$match": {
            "drop_3m": {"$ne": None}
            }
    },
    {
        "$group": {
            "_id": "$customer_id",
            "month_of_biggest_drop": {"$first": "$observation_month"},
            "biggest_drop": {"$first": "$drop_3m"}
            }
    }
    ]

results = collection_performance.aggregate(pipeline_q1)

for res in results:
    print (res)


#%% Running optimizer over query 1 version 1

q1_v1_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance",
        "pipeline" : pipeline_q1, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q1_v1_explain)

#%% 2. Pronaći klijente kod kojih je iskorišćenost kreditne kartice porasla više od 40%,
#  a kreditna sposobnost opala za više od 100 poena u prethodna 3 meseca pre likvidacije.

# potencijalne optimizacije:
# 1. sortiranje po has_defaulted, 0 unapred
# 2. sortiranje po mesecu observacije, asc


pipeline_q2 = [
    {
        '$match': {
            'dpd_current': {
                '$nin': [
                    0, 120
                ]
            }
        }
    }, {
        '$setWindowFields': {
            'partitionBy': {
                'customer_id': 1
            }, 
            'sortBy': {
                'customer_id': 1, 
                'observation_month': 1
            }, 
            'output': {
                'prevCredit': {
                    '$shift': {
                        'output': '$credit_score', 
                        'by': -3
                    }
                }, 
                'prevUtil': {
                    '$shift': {
                        'output': '$revolving_utilization', 
                        'by': -3
                    }
                }
            }
        }
    }, {
        '$addFields': {
            'creditDrops': {
                '$subtract': [
                    '$prevCredit', '$credit_score'
                ]
            }, 
            'utilDrops': {
                '$subtract': [
                    '$prevUtil', '$revolving_utilization'
                ]
            }
        }
    }, {
        '$addFields': {
            'finalizedDrops': {
                '$cond': {
                    'if': {
                        '$and': [
                            {
                                '$gt': [
                                    '$creditDrops', 100
                                ]
                            }, {
                                '$gt': [
                                    '$utilDrops', 0.4
                                ]
                            }
                        ]
                    }, 
                    'then': 'true', 
                    'else': 'false'
                }
            }
        }
    }, {
        '$match': {
            'finalizedDrops': 'true'
        }
    }
]

results = collection_performance.aggregate(pipeline_q2)

for res in results:
    print (res)
#%% Running optimizer over query 2 version 1

q2_v1_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance",
        "pipeline" : pipeline_q2, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q2_v1_explain)

#%% 3. Za korisnike sa lošim kreditnim rejtingom u trenutku odobrenja kredita (<579) odrediti 
# odnos ukupne sume izvršenih uplata i predviđene vrednosti uplata u prvih 12 meseci, kao i stopu defaulta.

# sum(payment_amount[:12]) / sum(scheduled_emi[:12])

# count(izvuceni.loc['has_defaulted' == True]) / count(izvuceni[:])

# ovo se radi nad origin skupom 
pipeline_q3 = [ 
    {
        '$match': {
            'credit_score_origination': {
                '$lte': 579
            }
        }
    }, {
        '$setWindowFields': {
            'partitionBy': {}, 
            'sortBy': {}, 
            'output': {
                'totalCount': {
                    '$count': {}, 
                    'window': {
                        'documents': [
                            'unbounded', 'unbounded'
                        ]
                    }
                }, 
                'defaultCount': {
                    '$sum': {
                        '$cond': {
                            'if': {
                                '$eq': [
                                    '$default_12m', 1
                                ]
                            }, 
                            'then': 1, 
                            'else': 0
                        }
                    }, 
                    'window': {
                        'documents': [
                            'unbounded', 'unbounded'
                        ]
                    }
                }
            }
        }
    }, {
        '$addFields': {
            'defaultRatio': {
                '$divide': [
                    '$defaultCount', '$totalCount'
                ]
            }
        }
    }, {
        '$lookup': {
            'from': 'monthly_performance', 
            'localField': 'customer_id', 
            'foreignField': 'customer_id', 
            'as': 'monthly_data'
        }
    }, {
        '$set': {
            'sumPaid': {
                '$sum': {
                    '$map': {
                        'input': {
                            '$slice': [
                                '$monthly_data', 12
                            ]
                        }, 
                        'as': 'item', 
                        'in': '$$item.payment_amount'
                    }
                }
            }, 
            'sumExpected': {
                '$sum': {
                    '$map': {
                        'input': {
                            '$slice': [
                                '$monthly_data', 12
                            ]
                        }, 
                        'as': 'item', 
                        'in': '$$item.scheduled_emi'
                    }
                }
            }
        }
    }, {
        '$addFields': {
            'paymentRatio': {
                '$cond': [
                    {
                        '$gt': [
                            '$sumExpected', 0
                        ]
                    }, {
                        '$divide': [
                            '$sumPaid', '$sumExpected'
                        ]
                                                                                                                   }, 0
                ]
            }
        }
    }                                               
]

#%%
results = collection_og.aggregate(pipeline_q3)

for res in results:
    print (res)

#%% Running optimizer over query 3 version 1

q3_v1_explain = db.command(
    "explain",
    {
        "aggregate" : "origination_data",
        "pipeline" : pipeline_q3, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q3_v1_explain)

#%% 4. Za svakog klijenta izračunati najduži uzastopni niz meseci bez propuštene uplate.

#note: this query can be disgustingly "optimized" for the current data:
# as the function for late payments took way too long to execute, it has been skipped and current db
# only introduces late payments exactly before default, we can get streak without any calculations whatsoever
# all we need to do is find the first time has_defaulted becomes 1 and return month_on_the_book - 3. if none is found, return # of rows user has

pipeline_q4 = [
    {
        '$group': {
            '_id': '$customer_id', 
            'payments': {
                '$push': '$payment_made'
            }
        }
    }, {
        '$project': {
            'longest_streak': {
                '$let': {
                    'vars': {
                        'result': {
                            '$reduce': {
                                'input': '$payments', 
                                'initialValue': {
                                    'current': 0, 
                                    'longest': 0
                                }, 
                                'in': {
                                    'current': {
                                        '$cond': [
                                            {
                                                '$eq': [
                                                    '$$this', 1
                                                ]
                                            }, {
                                                '$add': [
                                                    '$$value.current', 1
                                                ]
                                            }, 0
                                        ]
                                    }, 
                                    'longest': {
                                        '$max': [
                                            '$$value.longest', {
                                                '$cond': [
                                                    {
                                                        '$eq': [
                                                            '$$this', 1
                                                        ]
                                                    }, {
                                                        '$add': [
                                                            '$$value.current', 1
                                                        ]
                                                    }, 0
                                                ]
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }, 
                    'in': '$$result.longest'
                }
            }
        }
    }, {
        '$project': {
            '_id': 0, 
            'customer_id': '$_id', 
            'longest_streak': 1
        }
    }
]

results = collection_performance.aggregate(pipeline_q4)

for res in results:
    print (res)

#%% Running optimizer over query 4 version 1

q4_v1_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance",
        "pipeline" : pipeline_q4, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q4_v1_explain)


#%% 5.  Pronaći tri profila korisnika (starosnu grupu, rod, status zaposlenosti i opseg godišnjih prihoda) 
# koji imaju najmanji prosečan DTI i najveći procenat korisnnika sa urednihm uplatama tokom prvih 12 meseci otplate.

pipeline_q5 = [
    {
        '$addFields': {
            'age_group': {
                '$switch': {
                    'branches': [
                        {
                            'case': {
                                '$lt': [
                                    '$age', 25
                                ]
                            }, 
                            'then': '18-24'
                        }, {
                            'case': {
                                '$lt': [
                                    '$age', 35
                                ]
                            }, 
                            'then': '25-34'
                        }, {
                            'case': {
                                '$lt': [
                                    '$age', 45
                                ]
                            }, 
                            'then': '35-44'
                        }, {
                            'case': {
                                '$lt': [
                                    '$age', 55
                                ]
                            }, 
                            'then': '45-54'
                        }
                    ], 
                    'default': '55+'
                }
            }, 
            'income_range': {
                '$switch': {
                    'branches': [
                        {
                            'case': {
                                '$lt': [
                                    '$annual_income', 30000
                                ]
                            }, 
                            'then': '<30k'
                        }, {
                            'case': {
                                '$lt': [
                                    '$annual_income', 60000
                                ]
                            }, 
                            'then': '30k-60k'
                        }, {
                            'case': {
                                '$lt': [
                                    '$annual_income', 100000
                                ]
                            }, 
                            'then': '60k-100k'
                        }
                    ], 
                    'default': '100k+'
                }
            }
        }
    }, {
        '$group': {
            '_id': {
                'age_group': '$age_group', 
                'gender': '$gender', 
                'employment_status': '$employment_status', 
                'income_range': '$income_range'
            }, 
            'avg_dti': {
                '$avg': '$dti'
            }, 
            'regular_payment_percentage': {
                '$avg': {
                    '$cond': [
                        {
                            '$eq': [
                                '$default_12m', 0
                            ]
                        }, 100, 0
                    ]
                }
            }, 
            'customers': {
                '$sum': 1
            }
        }
    }, {
        '$sort': {
            'avg_dti': 1, 
            'regular_payment_percentage': -1
        }
    }, {
        '$limit': 3
    }
]

results = collection_og.aggregate(pipeline_q5)

for res in results:
    print (res)

#%% Running optimizer over query 5 version 1

q5_v1_explain = db.command(
    "explain",
    {
        "aggregate" : "origination_data",
        "pipeline" : pipeline_q5, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q5_v1_explain)


#%% rearanziranje baze - v2

#%% 1. korak - prosirena referenca

# prosirena referenca je basically niz vrednosti iz origination_data koji ce biti prebacen u performance dataframe
# e sad, problem malo nastaje jer se performance dataframe jako jako dugo pretrazuje jer ipak ima 2 miliona redova

#  Referenca bi se proširila
# da sadrži kreditni rejting u trenutku odobrenja kredita, rod, starosnu grupu i opseg prihoda korisnika
# (koji bi se dodatno morali preurediti tako da predstavljaju grupe umesto pojedniačnih vrednosti). 

def apply_extended_reference(df_credit_score, df_performance):
    origination_cols = [
    "age",
    "gender",
    "employment_status",
    "annual_income",
    "dti",
    "credit_score_origination"
    ]

    df_orig = df_credit_score.copy()

    df_orig["origination"] = df_orig[origination_cols].to_dict("records")

    df_orig = df_orig[["customer_id", "origination"]]

    df_result = df_performance.merge(
        df_orig,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    return df_result

#%% pozivanje funkcije

df_extended_ref = apply_extended_reference(df_credit_score, df_performance)

print (df_extended_ref.sample())
#%% 2. korak - precomputed 

def add_precomputed_fields(df):

    df["cumulative_payment_amount"] = (
        df.groupby("customer_id")["payment_amount"]
          .cumsum()
    )

    df["cumulative_scheduled_emi"] = (
        df.groupby("customer_id")["scheduled_emi"]
          .cumsum()
    )

    def consecutive_on_time(group):
        streak = 0
        result = []

        for dpd in group["dpd_current"]:
            if dpd == 0:
                streak += 1
            else:
                streak = 0

            result.append(streak)

        return pd.Series(result, index=group.index)

    df["consecutive_on_time_months"] = (
        df.groupby("customer_id", group_keys=False)
          .apply(consecutive_on_time)
    )

    df["credit_score_delta_3m"] = (
        df.groupby("customer_id")["credit_score"].shift(3) - df["credit_score"]
    )

    df["revolving_utilization_delta_3m"] = (
        df.groupby("customer_id")["revolving_utilization"].shift(3) -    df["revolving_utilization"]
        )

    return df

#%% pozivanje precomputed 

df_extended_ref = add_precomputed_fields(df_extended_ref)

print (df_extended_ref.columns)


#%% 3. korak, refill the db!

db.monthly_performance_v2.insert_many(
    df_extended_ref.to_dict("records")
)

print ("Insert complete!")

collection_performance_v2 = db['monthly_performance_v2']
#%% 4. korak indexi

# customer_id ista prica od ranije, neka bude u kombinaciji sa month_on_book za neke hronoloske stvari
# { customer_id: 1, months_on_book: 1 }

# za query 3 i sada vec discarded query 4, origination credit score moze da dosta ubrza matching 
# { "origination.credit_score_origination": 1 }

collection_performance_v2.create_index(
    [("customer_id", 1), ("months_on_book", 1)],
    name="customer_mob_idx"
)

collection_performance_v2.create_index(
    [("origination.credit_score_origination", 1)],
    name="origination_score_idx"
)

print("Indexes created!")



#%% query 1 ver 2

# 1. Za svakog klijenta koji je ušao u default, odrediti mesec pre ulaska u 
# default u kom je imao najveći pad kreditnog rejtinga u odnosu na 3 meseca ranije.

pipeline_q1_v2 = [
    {
        '$match': {
            'has_defaulted': 1, 
            'credit_score_delta_3m': {
                '$ne': None
            }
        }
    }, {
        '$sort': {
            'customer_id': 1, 
            'credit_score_delta_3m': 1
        }
    }, {
        '$group': {
            '_id': '$customer_id', 
            'month_of_biggest_drop': {
                '$first': '$observation_month'
            }, 
            'biggest_drop': {
                '$first': '$credit_score_delta_3m'
            }
        }
    }
]

results = collection_performance_v2.aggregate(pipeline_q1_v2)

for res in results:
    print (res)


#%% Running optimizer over query 1 ver 2

q1_v2_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance_v2",
        "pipeline" : pipeline_q1_v2, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q1_v2_explain)

#%% q2v2 

pipeline_q2_v2 = [
    {
        '$match': {
            'dpd_current': {
                '$nin': [
                    0, 120
                ]
            }
        }
    }, {
        '$addFields': {
            'finalizedDrops': {
                '$cond': {
                    'if': {
                        '$and': [
                            {
                                '$gt': [
                                    '$credit_score_delta_3m', 100
                                ]
                            }, {
                                '$gt': [
                                    '$revolving_utilization_delta_3m', 0.4
                                ]
                            }
                        ]
                    }, 
                    'then': 'true', 
                    'else': 'false'
                }
            }
        }
    }, {
        '$match': {
            'finalizedDrops': 'true'
        }
    }
]


results = collection_performance_v2.aggregate(pipeline_q2_v2)

for res in results:
    print (res)


#%% Running optimizer over query 2 ver 2

q2_v2_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance_v2",
        "pipeline" : pipeline_q2_v2, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q2_v2_explain)


#%% q3v2 

#%% q4v2

pipeline_q4_v2 = [
    {
        '$group': {
            '_id': '$customer_id', 
            'maxConsistency': {
                '$max': '$consecutive_on_time_months'
            }
        }
    }
]


results = collection_performance_v2.aggregate(pipeline_q4_v2)

for res in results:
    print (res)


#%% Running optimizer over query 4 ver 2

q4_v2_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance_v2",
        "pipeline" : pipeline_q4_v2, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q4_v2_explain)

#%% q5v2

pipeline_q5_v2 = [
    {
        '$addFields': {
            'age_group': {
                '$switch': {
                    'branches': [
                        {
                            'case': { '$lt': [ '$origination.age', 25 ] },
                            'then': '18-24'
                        },
                        {
                            'case': { '$lt': [ '$origination.age', 35 ] },
                            'then': '25-34'
                        },
                        {
                            'case': { '$lt': [ '$origination.age', 45 ] },
                            'then': '35-44'
                        },
                        {
                            'case': { '$lt': [ '$origination.age', 55 ] },
                            'then': '45-54'
                        }
                    ],
                    'default': '55+'
                }
            },

            'income_range': {
                '$switch': {
                    'branches': [
                        {
                            'case': { '$lt': [ '$origination.annual_income', 30000 ] },
                            'then': '<30k'
                        },
                        {
                            'case': { '$lt': [ '$origination.annual_income', 60000 ] },
                            'then': '30k-60k'
                        },
                        {
                            'case': { '$lt': [ '$origination.annual_income', 100000 ] },
                            'then': '60k-100k'
                        }
                    ],
                    'default': '100k+'
                }
            }
        }
    },
    {
        '$group': {
            '_id': {
                'age_group': '$age_group',
                'gender': '$origination.gender',
                'employment_status': '$origination.employment_status',
                'income_range': '$income_range'
            },
            'avg_dti': {
                '$avg': '$origination.dti'
            },
            'regular_payment_percentage': {
                '$avg': {
                    '$cond': [
                        { '$eq': [ '$will_default_original', 0 ] },
                        100,
                        0
                    ]
                }
            },
            'customers': {
                '$sum': 1
            }
        }
    },
    {
        '$sort': {
            'avg_dti': 1,
            'regular_payment_percentage': -1
        }
    },
    {
        '$limit': 3
    }
]


results = collection_performance_v2.aggregate(pipeline_q5_v2)

for res in results:
    print (res)


#%% Running optimizer over query 2 ver 2

q5_v2_explain = db.command(
    "explain",
    {
        "aggregate" : "monthly_performance_v2",
        "pipeline" : pipeline_q5_v2, 
        "cursor" : {}
    },
    verbosity = "executionStats"
)

pprint.pprint(q5_v2_explain)
