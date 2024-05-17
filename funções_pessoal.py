import numpy as np
import pandas as pd
import math
import datetime as dt
import requests

def analise_de_preços(df, n_vol):
    '''retorna
    df = dataframe de cotas;'''
    dict_stats = {}
    for col in df.columns:
        dict_stats[f'media {col}'] = df[col].mean()
        dict_stats[f'vol {col}'] = df[col].std()
        dict_stats[f'm + vol {col}'] = dict_stats[f'media {col}'] + dict_stats[f'vol {col}']
        dict_stats[f'm - vol {col}'] = dict_stats[f'media {col}'] - dict_stats[f'vol {col}']
        dict_stats[f'm + {n_vol}vol {col}'] = dict_stats[f'media {col}'] + n_vol * dict_stats[f'vol {col}']
        dict_stats[f'm - {n_vol}vol {col}'] = dict_stats[f'media {col}'] - n_vol * dict_stats[f'vol {col}']

    tabela = pd.DataFrame()
    for col in df.columns:
        tabela.loc[f'm + {n_vol}vol', col] = dict_stats[f'm + {n_vol}vol {col}']
        tabela.loc[f'm + vol', col] = dict_stats[f'm + vol {col}']
        tabela.loc[f'Média', col] = dict_stats[f'media {col}']
        tabela.loc[f'm - vol', col] = dict_stats[f'm - vol {col}']
        tabela.loc[f'm - {n_vol}vol', col] = dict_stats[f'm - {n_vol}vol {col}']
    
    return dict_stats, tabela

def obter_dados_bcb_sgs(codigo_serie):
    '''A função pega dados do sistema de dados do BCB e retorna um dataframe;
    codigo_serie = código referente à série de dados que deseja-se obter os dados'''
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        # Converte os dados para um DataFrame
        dados = pd.DataFrame(response.json())
        dados['valor'] = pd.to_numeric(dados['valor'])
        dados['data'] = pd.to_datetime(dados['data'], format='%d/%m/%Y')
        return dados
    else:
        print("Falha na requisição: ", response.status_code)
        return None
    
def obter_dados_bcb_sgs_v2(codigo_serie):
    '''A função pega dados do sistema de dados do BCB e retorna um dataframe;
    codigo_serie = código referente à série de dados que deseja-se obter os dados'''
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_serie}/dados?formato=json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            # Converte os dados para um DataFrame
            dados = pd.DataFrame(response.json())
            dados['valor'] = pd.to_numeric(dados['valor'])
            dados['data'] = pd.to_datetime(dados['data'], format='%d/%m/%Y')
            return dados
        except ValueError as e:
            print("Falha na decodificação JSON: ", e)
            return None
    else:
        print("Falha na requisição: ", response.status_code, response.text)
        return None