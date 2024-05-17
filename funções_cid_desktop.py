import numpy as np
import pandas as pd
import math
import datetime as dt

# criando fórmula
def gerar_multiplos(num, quantidade):
    lista_multiplos = []
    for i in range(0, quantidade):
        lista_multiplos.append(i * num)
    return lista_multiplos

def formatar_porcentagem(valor, posição):
    return '{:.0%}'.format(valor)

def retorno_acumulado(df):
    first_line = df.iloc[0]
    ret_acum = df.divide(first_line) - 1
    return ret_acum

def retorno_acumulado_não_normalizado(df):
    first_line = df.iloc[0]
    ret_acum = df.divide(first_line)
    return ret_acum

def retorno_diario(df, method = None):
    '''df = dataframe de cotas
method = define o método de calculo
(0 para deixar primeira linha igual a zero)
(1 para deixar a primeira linha com NaN)'''
    if method == 0 or method == None:
        retornos = df.pct_change()
        retornos.iloc[0] = 0
        return retornos
    else:
        retornos = df.pct_change()
        return retornos
    
def retorno_rolling(df_cotas, janela):
    df = df_cotas.copy()
    ret_roll = df / df.shift(janela) - 1
    return ret_roll

def anualizar_taxa(df_retornos):
    '''Retorna um dataframe com as taxas inputadas anualizadas;
    df_retornos = dataframe de input de retornos ou de qualquer tipo de taxa'''
    primeiras_datas_validas = df_retornos.apply(pd.Series.first_valid_index)
    taxa_anual = pd.DataFrame(index = df_retornos.index)
    for i in df_retornos.columns:
        df_sample = pd.DataFrame()
        data = primeiras_datas_validas[i]
        df_sample[i] = df_retornos.loc[data: , i]
        taxa_anual[i] = (1 + df_sample) ** (252/len(df_sample)) - 1
    return taxa_anual

def diarizar_taxa_anualizada(taxa_anualizada):
    return ((1 + taxa_anualizada) ** (1/252)) - 1

def retorno_acumulado_cumprod(df, df_type, input_type = None):
    ''' Retorna um dataframe de retorno acumulado utilizando o método cumprod;
    df = dataframe de cotas ou retornos diarios;
    df_type = 0 (df de input é de cotas);
    df_type = 1 (df de input é de retornos diários);
    input_type = se tiver NaN no df escrever 'with nan' se não tiver NaN, não preencher'''
    if input_type == 'with nan':
        if df_type == 0:
            df_retornos = retorno_diario(df, 1)
            ret_acum = (1 + df_retornos).cumprod() - 1
            return ret_acum
        if df_type == 1:
            ret_acum = (1 + df).cumprod() - 1
            return ret_acum
    else:
        if df_type == 0:
            df_retornos = retorno_diario(df, 1)
            ret_acum = (1 + df_retornos).cumprod() - 1
            return ret_acum
        if df_type == 1:
            ret_acum = (1 + df).cumprod() - 1
            return ret_acum

def retorno_acumulado_anualizado(df):
    '''A função retorna um dataframe com os retornos acumulados anualizados;
    df = dataframe de cotas'''
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    ret_acum_anual = pd.DataFrame(index = df.index)
    for i in df.columns:
        df_sample = pd.DataFrame()
        data = primeiras_datas_validas[i]
        df_sample[i] = df.loc[data: , i]
        ret_acum = retorno_acumulado_cumprod(df_sample, df_type=0)
        df_sample = df_sample.reset_index()
        for index, date in zip(df_sample.index, df_sample['Date']) :
            ret_acum_anual.loc[date, i] = (1 + ret_acum.loc[date, i]) ** (252/(index + 1)) - 1
    return ret_acum_anual

def df_datas_desejadas(df, dia_inicial, dia_final):
    df = df.loc[dia_inicial : dia_final]
    return df

def tratamento_benchmarks(df):
    lista = [0, 1, 2]
    for i in lista:
        df = df.drop(i)
    #df = df.bfill()
    #df = df.dropna()
    df = df.set_index('Date')
    return df

def nick_maker(df_ou_lista, input_type):
    '''Cria um dicionário de nicknames para as colunas de um df
    de modo a poder utilizar esses nicknames como variáveis'''
    if input_type == 'dataframe':
        lista = []
        for i in df_ou_lista.columns:
            lista.append(i)
        lista = sorted(lista)
        texto = ','.join(lista)
        texto = texto.lower()
        out = ['+', '-', '*', '=']
        for i in out:
            texto = texto.replace(i, '_plus')
        texto = texto.replace(' ', '_')
        nicks = texto.split(',')
        dict_nicks = {}
        for i, col in zip(nicks, lista):
            dict_nicks[col] = i
        return dict_nicks
    elif input_type == 'lista':
        lista = []
        for i in df_ou_lista:
            lista.append(i)
        lista = sorted(lista)
        texto = ','.join(lista)
        texto = texto.lower()
        out = ['+', '-', '*', '=']
        for i in out:
            texto = texto.replace(i, '_plus')
        texto = texto.replace(' ', '_')
        nicks = texto.split(',')
        dict_nicks = {}
        for i, col in zip(nicks, lista):
            dict_nicks[col] = i
        return dict_nicks

def lista_maker(tornar_lista):
    '''Cria uma lista com as colunas ou com o index de um dataframe.'''
    lista = []
    for i in tornar_lista:
        lista.append(i)
    return lista

def add_lista(lista1, lista2):
    '''Adiciona os elementos da lista1 na lista 2'''
    for i in lista1:
        lista2.append(i)

def tirar_fds(dict_data_tirar, df):
    '''Tira as datas que são == 0 do dicionário feito com base em count()'''
    for i in df.index:
        if dict_data_tirar[i] == 0:
            df = df.drop(i)
        else:
            pass
    return df

def anos_agrupados(df):
    '''A função retona dfs '''
    year_group = df.groupby(df.index.year)

    first_year = df.index[0].year
    last_year = df.index[-1].year

    anos = []

    for i in range(first_year, last_year + 1):
        anos.append(i)
        globals()['cotas_' + str(i)] = year_group.get_group(i)

def performance_anualizada_por_ano(df, output_type):
    '''A função retorna um df com o retorno de cada ano anualizado 
    do dataframe inputado;
    df = df de cotas/preços;
    output_type = define se é para ser um df funcional 
    (sendo possível realizar contas/output_type = 0) ou um df não funcional 
    (apenas apresentável/output_type = 1)'''
    year_group = df.groupby(df.index.year)
    first_year = df.index[0].year
    last_year = df.index[-1].year

    anos = []

    dict = {}

    for i in range(first_year, last_year + 1):
        anos.append(i)
        dict['cotas_' + str(i)] = year_group.get_group(i)

    print(f'dataframes separados com dados dos seguintes anos: {anos} (exemplo: cotas_{i})')

    aux = dict[f'cotas_{first_year}']
    df_desempenho = pd.DataFrame(columns = aux.columns)

    for i in range(first_year, last_year + 1):
        # Retorno diário de cada ano
        df_aux = dict[f'cotas_{i}']

        # Retorno diário de cada ano
        dict[f'retornos_{i}'] = retorno_diario(df_aux)
        df_aux_retornos = dict[f'retornos_{i}']

        # Retorno Acumulado de cada ano:
        dict[f'ret_acum_{i}'] = retorno_acumulado(df_aux)
        df_aux_ret_acum = dict[f'ret_acum_{i}']

        # Preenchendo o df_desempenho com o retorno acumulado de cada ano
        df_desempenho.loc[i] = (1 + df_aux_ret_acum.iloc[-1]) ** (252/len(df_aux_ret_acum)) - 1
    if output_type == 1:
        df_desempenho = df_desempenho.applymap(lambda x: "{:.2%}".format(x))
    else:
        pass
    return df_desempenho  

#Volatilidade Anualizada
def volatilidade_anualizada(df):
    '''df = df de cotas'''
    df_retornos = retorno_diario(df)
    df_vol = df_retornos.std()
    df_vol_anualizada = df_vol * (252) ** (1/2)
    return df_vol_anualizada

#Downside Risk Anualizado
def downside_risk_anualizado(df):
    '''Retorna uma série com o valor do downside risk dos ativos;
    df = dataframe de cotas'''
    df_retornos = retorno_diario(df)
    df_retornos_negativos = df_retornos[df_retornos < 0]
    df_vol = df_retornos_negativos.std()
    df_vol_anualizada = df_vol * (252) ** (1/2)
    return df_vol_anualizada

#Performance Por Mês
def performance_por_mes(df_retornos, output_type):
    '''Retorna um dataframe do retorno acumulado de cada mês;
    df_retornos = dataframe de retornos diários;
    output_type = define se é para ser um df funcional 
    (sendo possível realizar contas/output_type = 0) 
    ou um df não funcional (apenas apresentável/output_type = 1)'''
    #df_retornos = retorno_diario(df_ativo)
    df = df_retornos.copy()
    df['ano'] = df.index.year
    df['mes'] = df.index.month
    grouped = df.groupby(['ano', 'mes'])

    dict = {}
    df_fill = pd.DataFrame(columns = df.columns)
    for i in df.index:
        ano = df.loc[i, 'ano']
        mes = df.loc[i, 'mes']
        dict[f'{ano}_{mes}'] = grouped.get_group((ano, mes))
        df_aux = dict[f'{ano}_{mes}']
        
        # Retorno Acumulado
        df_aux_ret_acum = retorno_acumulado_cumprod(df_aux, df_type = 1)

        # Preenchendo o df
        df_fill.loc[f'{mes}/{ano}'] = df_aux_ret_acum.iloc[-1]
    
    df_fill = df_fill.drop('ano', axis = 1)
    df_fill = df_fill.drop('mes', axis = 1)

    if output_type == 1:
        df_fill = df_fill.applymap(lambda x: "{:.2%}".format(x))
    else:
        pass
    return df_fill

#Rentabilidade em Períodos Definidos 
def data_escolha(df, data_inicio, data_fim):
    df_aux = df.loc[data_inicio : data_fim]
    return df_aux
def rentabilidade_data_escolha(df_cotas, data_inicio, data_fim, fundo=None):
    df = df_cotas.loc[data_inicio : data_fim]
    df_ret_acum = retorno_acumulado_cumprod(df, df_type = 0)
    last_line = df_ret_acum.iloc[-1]
    if fundo == None:
        return last_line
    else:
        resp = last_line[fundo]
        return resp

#Segmentação de dfs por ano
def df_por_ano(df, output_type):
    '''Retorna um dicionário com dataframes de dados por ano'''

    year_group = df.groupby(df.index.year)

    first_year = df.index[0].year
    last_year = df.index[-1].year

    anos = []
    dict = {}

    for i in range(first_year, last_year + 1):
        anos.append(i)
        dict[i] = year_group.get_group(i)
    
    if output_type == 'dict':
        return dict
    else:
        return anos

#Check NaN de dfs de Correlação
def check_nan_corr(df):
    coluna1 = df.columns[0]
    for i in df.index:
        valor = df.loc[i, coluna1]
        if math.isnan(valor):
            df = df.drop(i)
        else:
            pass
    for col in df.columns:
        valor = df.loc[coluna1, col]
        if math.isnan(valor):
            df = df.drop(col, axis = 1)
        else:
            pass
    return df

def mudar_nome_coluna(df, nomes_col_nova):
    '''A função retorna um novo df com as colunas do df renomeadas;
    df = dataframe que se deseja renomear as colunas;
    nomes_col_nova = lista com os novos nomes das colunas;
    # OBS as colunas do df e a lista nomes_col_nova precisam estar na mesma ordem
    '''
    for col_antiga, col_nova in zip(df.columns, nomes_col_nova):
        df = df.rename(columns = {col_antiga: col_nova})
    return df

def most_recent_valid_index_df(df):
    '''Retorna um df que se inicia na linha do index mais recente;
    df = df de cotas ou retornos, necessariamente com NaN'''
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    data = primeiras_datas_validas.max()
    df = df.loc[data:]
    return df 

## Funções Métricas

#Quintil benchmarks
def quintil_bench(bench_analise):
    '''bench_analise = df de retornos diários dos benchmarks.
    A função retorna um df com os quintis do df bench_analise fornecido'''
    lista_index_quartil = ['q1', 'q2', 'q3', 'q4', 'q5']
    quartil_bench = pd.DataFrame(index=lista_index_quartil)
    quartil_datas = pd.DataFrame(index=lista_index_quartil)

    for bench in bench_analise.columns:
        q1 = np.percentile(bench_analise[bench], 20)
        q2 = np.percentile(bench_analise[bench], 40)
        q3 = np.percentile(bench_analise[bench], 60)
        q4 = np.percentile(bench_analise[bench], 80)
        q5 = np.percentile(bench_analise[bench], 99.5)
        valor_proximo_q1 = bench_analise[bench].iloc[(bench_analise[bench] - q1).abs().argsort()[:1]].item()
        valor_proximo_q2 = bench_analise[bench].iloc[(bench_analise[bench] - q2).abs().argsort()[:1]].item()
        valor_proximo_q3 = bench_analise[bench].iloc[(bench_analise[bench] - q3).abs().argsort()[:1]].item()
        valor_proximo_q4 = bench_analise[bench].iloc[(bench_analise[bench] - q4).abs().argsort()[:1]].item()
        valor_proximo_q5 = bench_analise[bench].iloc[(bench_analise[bench] - q5).abs().argsort()[:1]].item()
        # preenchendo o df quartil com os quartis calculados
        quartil_bench.loc['q1', bench] = q1
        quartil_bench.loc['q2', bench] = q2
        quartil_bench.loc['q3', bench] = q3
        quartil_bench.loc['q4', bench] = q4
        quartil_bench.loc['q5', bench] = q5
        # preenchendo o df quartil_datas com as datas de cada quartil no df
        for date in bench_analise.index:
            if bench_analise.loc[date, bench] == valor_proximo_q1:
                data_q1 = date
                quartil_datas.loc['q1', bench] = data_q1
            elif bench_analise.loc[date, bench] == valor_proximo_q2:
                data_q2 = date
                quartil_datas.loc['q2', bench] = data_q2
            elif bench_analise.loc[date, bench] == valor_proximo_q3:
                data_q3 = date
                quartil_datas.loc['q3', bench] = data_q3
            elif bench_analise.loc[date, bench] == valor_proximo_q4:
                data_q4 = date
                quartil_datas.loc['q4', bench] = data_q4
            elif bench_analise.loc[date, bench] == valor_proximo_q5:
                data_q5 = date
                quartil_datas.loc['q5', bench] = data_q5
            else:
                pass
            
    return quartil_bench

#Quintil fundos
def quintil_fundos(retornos_fundos):
    '''retornos_fundos = df de retornos diários dos fundos ou ativos
    A função retorna um df dos quintis do df retornos_fundos fornecido'''
    lista_index_quartil = ['q1', 'q2', 'q3', 'q4', 'q5']
    quartil_fundos = pd.DataFrame(index=lista_index_quartil)
    for fund in retornos_fundos.columns:
        q1 = np.percentile(retornos_fundos[fund], 20)
        q2 = np.percentile(retornos_fundos[fund], 40)
        q3 = np.percentile(retornos_fundos[fund], 60)
        q4 = np.percentile(retornos_fundos[fund], 80)
        q5 = np.percentile(retornos_fundos[fund], 100)
        # preenchendo o df quartil com os quartis calculados
        quartil_fundos.loc['q1', fund] = q1
        quartil_fundos.loc['q2', fund] = q2
        quartil_fundos.loc['q3', fund] = q3
        quartil_fundos.loc['q4', fund] = q4
        quartil_fundos.loc['q5', fund] = q5
    
    return quartil_fundos

#Média Quintil
def media_quantil(retornos_fundos, bench_analise, quantil, output_type, fundo_foco=None):
    '''retornos_fundos = df de retornos diários dos fundos
    bench_analise = df de retornos diários dos benchmarks
    quantil = define o tipo de percentil desejado (5 = quintil, 4 = quartil, ...)
    output_type = define qual o df é para ser retornado ou o de fundo ou o de benchmarks
    fundo_foco = define qual fundo é para gerar a média de quantis (se output_type = 'fundo', esse parâmetro precisa ser definido) '''
    
    # 0) Organizando o df inputado
    bench_analise = bench_analise.reindex(sorted(bench_analise.columns), axis = 1) #deixando colunas do df em ordem alfabética
    dict_nicks = nick_maker(bench_analise) # Criando nicknames para as colunas do df
    dict_nicks_fundos = nick_maker(retornos_fundos)
    dict_nicks.update(dict_nicks_fundos)
    print(dict_nicks)
    lista_index_quantil = [] # Criando a lista que será utilizada como index do df final de média
    for i in range(1, quantil + 1):
        texto = 'q' + str(i)
        lista_index_quantil.append(texto)
    print(lista_index_quantil)

    # 1) Criando dataframes diferentes para cada bench, ou seja, um df novo por coluna
    for col in bench_analise.columns:
        nome = dict_nicks[col]
        globals()[nome] = pd.DataFrame(bench_analise[col])

    # 2) Organizando os benchmarks por ordem crescente dos valores de retorno (indo do mais negativo para o mais positivo)
    for col in bench_analise.columns:
        nick = dict_nicks[col]
        globals()[nick] = globals()[nick].sort_values(by = col)
        #print(globals()[nick])

    n = len(bench_analise)//quantil # DIVISOR: como é quintil, é necessário dividir a qntd de linhas do df por 5. Como todos os dfs têm a mesma quantidade de linhas, é possível escolher qualquer um, mas tem que adaptar o código primeiro
    
    # 3) Calculando a média de intervalos de 20% dos dados (média do intervalo de cada quintil)
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        globals()['medias_' + bench] = globals()[bench].rolling(window=n, min_periods=n, step=n).mean(numeric_only = True)
        globals()['medias_' + bench] = globals()['medias_' + bench].reset_index()
        globals()['medias_' + bench] = globals()['medias_' + bench].drop('Date', axis=1)
        print(globals()['medias_' + bench])
    
    # 4) As médias foram calculadas, mas a média do último intervalo não consta e o df das médias encontra-se com NaN. Logo essa parte do código calcula a média dos primeiros 20%
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        if globals()['medias_' + bench].isna().any().any(): # avalia a presença de NaN no df medias_hfri, por exemplo, se houver NaN um tratamento é feito. Se não há NaNs o nenhum tratamento é realizado.
            globals()[bench + '_aux'] = globals()[bench].copy()
            globals()[bench + '_aux'] = globals()[bench + '_aux'].reset_index()
            media = globals()[bench + '_aux'].tail(n).mean(numeric_only = True) # pegando os últimos dados que representam 20% do total de dados e são a ponta dos maiores retornos e calculando a média
            globals()['medias_' + bench] = globals()['medias_' + bench].dropna() #tirando a linha com NaN
            globals()['medias_' + bench].loc[5] = media # preenchendo os dfs com as médias calculadas da ponta final dos dataframes 
        else:
            print('Check OK: não possui NaN')
        print(globals()['medias_' + bench])
    
    # 5) Criando um df com as médias de cada benchmark que foram calculadas separadamente em (3)
    medias_bench = pd.DataFrame()
    for col in bench_analise.columns:
        nick = dict_nicks[col]
        medias_bench[col] = globals()['medias_' + nick][col]
    medias_bench = medias_bench.set_index(pd.Index(lista_index_quantil)) # Um dos dfs que pode ser retornado, dependendo do output desejado
    
    # 6) Criando uma lista para cada um dos dfs de forma a salvar a ordem de cada uma das séries de datas (index) dos dfs dos benchmarks
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        globals()[bench + '_index_lista'] = []
        for index in globals()[bench].index:
            globals()[bench + '_index_lista'].append(index)
        print(globals()[bench + '_index_lista'])

    # 7) Organizando o df de retornos dos fundos (retornos_fundos) seguindo a ordem das listas geradas acima
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        globals()['fundos_index_' + bench] = retornos_fundos.copy()
        globals()['fundos_index_' + bench] = globals()['fundos_index_' + bench].reindex(pd.to_datetime(globals()[bench + '_index_lista']))
        print(globals()['fundos_index_' + bench])

    # 8) Calculando a média dos intervalos dos quantis dos dados de retornos dos fundos (média do intervalo de cada quantil)
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        globals()['medias_fundos_' + bench] = globals()['fundos_index_' + bench].rolling(window=n, min_periods=n, step=n).mean(numeric_only = True)
        globals()['medias_fundos_' + bench] = globals()['medias_fundos_' + bench].reset_index()
        globals()['medias_fundos_' + bench] = globals()['medias_fundos_' + bench].drop('index', axis=1)
        print(globals()['medias_fundos_' + bench])
    
    # 9) As médias foram calculadas, mas a média do último intervalo não consta e o df das médias encontra-se com NaN
    for col in bench_analise.columns:
        bench = dict_nicks[col]
        if globals()['medias_fundos_' + bench].isna().any().any(): # avalia a presença de NaN no df medias_fundos_hfri, por exemplo, se houver NaN um tratamento é feito. Se não há NaNs, nenhum tratamento é realizado.
            globals()['fundos_index_aux_' + bench] = globals()['fundos_index_' + bench].copy()
            media = globals()['fundos_index_aux_' + bench].tail(n+1).mean(numeric_only = True) # pegando os últimos dados que representam 20% do total de dados e são a ponta dos maiores retornos e calculando a média
            globals()['medias_fundos_' + bench] = globals()['medias_fundos_' + bench].dropna() #tirando a linha com NaN
            globals()['medias_fundos_' + bench].loc[5] = media # preenchendo os dfs com as médias calculadas da ponta final dos dataframes 
        else:
            print('Check OK: não possui NaN')
        print(globals()['medias_fundos_' + bench])
    
    # 10) Criando dfs com as médias de cada fundo em relação a cada benchmark que foram calculadas separadamente em (8)
    if fundo_foco != None:
        fund = dict_nicks[fundo_foco]
        globals()['medias_' + fund] = pd.DataFrame() # definir o df vazio e sem nenhum index pq o df medias_defensive e medias_hfri iriam possuir index diferentes, o que iria dar ruim.
        for col in bench_analise.columns:
            bench = dict_nicks[col]
            globals()['medias_' + fund][col] =  globals()['medias_fundos_' + nick][fundo_foco]
            globals()['medias_' + fund] = globals()['medias_' + fund].rename(columns={f'{col}': f'{col} x {fundo_foco}'})
        globals()['medias_' + fund] = globals()['medias_' + fund].set_index(pd.Index(lista_index_quantil))
        print(globals()['medias_' + fund])

    if output_type == 'fundo':
        return globals()['medias_' + fund]
    
    elif output_type == 'benchmarks':
        return medias_bench

# Rolling Vol
def rolling_volatilidade(df, df_type, n):
    '''A função retorna o df de volatilidade rolling de n dias;
    df = dataframe utilizado para calcular vol rolling (pode ser de cotas ou de retornos diários);
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    n = número de dias para o rolling
    EX: df_roll_vol = rolling_volatilidade_anualizada(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    if df_type == 'cota':
        retornos_df = retorno_diario(df, 1)
        roll_vol = retornos_df.rolling(n).std()
        return roll_vol
    elif df_type == 'retorno':
        roll_vol = df.rolling(n).std()
        return roll_vol

#Rolling Vol Anualizada
def rolling_volatilidade_anualizada(df, df_type, n):
    '''A função retorna o df de volatilidade rolling de n dias;
    df = dataframe utilizado para calcular vol rolling (pode ser de cotas ou de retornos diários);
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    n = número de dias para o rolling
    EX: df_roll_vol = rolling_volatilidade_anualizada(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    if df_type == 'cota':
        retornos_df = retorno_diario(df, 1)
        roll_vol = retornos_df.rolling(n).std()
        roll_vol_anualizado = roll_vol * (252) ** (1/2)
        return roll_vol_anualizado
    elif df_type == 'retorno':
        roll_vol = df.rolling(n).std()
        roll_vol_anualizado = roll_vol * (252) ** (1/2)
        return roll_vol_anualizado
    
# Volatilidade Diária
def volatilidade_diaria(df):
    '''Retorna um dataframe com a volatilidade diária
    df = dataframe de cotas'''
    # 1) Cálculo do retorno diário 
    df_retornos = retorno_diario(df, 1)

    # 2) Cálculo da vol diária
    vol_diaria = pd.DataFrame(columns = df_retornos.columns, index = df_retornos.index)
    for col in df_retornos.columns:
        for index in range(len(df_retornos)):
            vol_diaria.iloc[index][col] = df_retornos.iloc[:index + 1][col].std()

    return vol_diaria

# Volatilidade Diária Anualizada
def volatilidade_diaria_anualizada(df):
    '''Retorna um dataframe com as volatilidades diárias anualizadas;
    df = dataframe de cotas'''
    # 1) Cálculo do retorno diário 
    df_retornos = retorno_diario(df, 1)

    # 2) Cálculo da vol diária
    vol_diaria = pd.DataFrame(columns = df_retornos.columns, index = df_retornos.index)
    for col in df_retornos.columns:
        for index in range(len(df_retornos)):
            vol_diaria.iloc[index][col] = df_retornos.iloc[:index + 1][col].std()

    # 3) Anualizando a vol diária
    vol_diaria_anualizada = vol_diaria * (252) ** (1/2)

    return vol_diaria_anualizada

# Downside Risk

# Downside risk diário
def downside_risk_diario(df):
    '''Retorna um dataframe com o downside risk ao longo dos dias;
    df = dataframe de cotas'''

    # 1) Cálculo do retorno diário 
    df_retornos = retorno_diario(df, 1)

    # 2) Retornos Negativos
    df_retornos_negativos = df_retornos[df_retornos < 0]

    # 3) Cálculo da vol diária
    vol_diaria = pd.DataFrame(columns = df_retornos_negativos.columns, index = df_retornos_negativos.index)
    for col in df_retornos_negativos.columns:
        for index in range(len(df_retornos_negativos)):
            vol_diaria.iloc[index][col] = df_retornos_negativos.iloc[:index + 1][col].std()
    
    return vol_diaria

# Downside risk diário anualizado
def downside_risk_diario_anualizado(df):
    # 1) Cálculo do retorno diário 
    df_retornos = retorno_diario(df, 1)

    # 2) Retornos Negativos
    df_retornos_negativos = df_retornos[df_retornos < 0]

    # 2) Cálculo da vol diária
    vol_diaria = pd.DataFrame(columns = df_retornos_negativos.columns, index = df_retornos_negativos.index)
    for col in df_retornos_negativos.columns:
        for index in range(len(df_retornos_negativos)):
            vol_diaria.iloc[index][col] = df_retornos_negativos.iloc[:index + 1][col].std()

    # 3) Anualizando a vol diária
    vol_diaria_anualizada = vol_diaria * (252) ** (1/2)

    return vol_diaria_anualizada

def rolling_downside_risk(df, df_type, n, min_valores_not_nan):
    '''A função retorna o df de downside risk rolling de n dias com datas casadas;
    df = dataframe utilizado para calcular vol rolling (pode ser de cotas ou de retornos diários);
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    n = número de dias para o rolling;
    min_valores_nao_nan = qntd mínimima de valores não nan para fazer o calculo da métrica;
    EX: df_roll_vol = rolling_volatilidade(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    if df_type == 'cota':
        retornos_df = retorno_diario(df, 1)
        df_ret_negativo = retornos_df[retornos_df < 0]
        roll_downside = df_ret_negativo.rolling(n, min_periods =  min_valores_not_nan).std()
        return roll_downside
    elif df_type == 'retorno':
        df_ret_negativo = df[df < 0]
        roll_downside = df_ret_negativo.rolling(n, min_periods =  min_valores_not_nan).std()
        return roll_downside
    
def rolling_downside_risk_anualizado(df, df_type, n, min_valores_not_nan):
    '''A função retorna o df de downside risk rolling anualizado de n dias com datas casadas;
    df = dataframe utilizado para calcular vol rolling (pode ser de cotas ou de retornos diários);
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    n = número de dias para o rolling;
    min_valores_nao_nan = qntd mínimima de valores não nan para fazer o calculo da métrica;
    EX: df_roll_vol = rolling_volatilidade(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    if df_type == 'cota':
      retornos_df = retorno_diario(df, 1)
      df_ret_negativo = retornos_df[retornos_df < 0]
      roll_downside = df_ret_negativo.rolling(n, min_periods = min_valores_not_nan).std() * (252) ** (1/2)
      return roll_downside
    elif df_type == 'retorno':
      df_ret_negativo = df[df < 0]
      roll_downside = df_ret_negativo.rolling(n, min_periods = min_valores_not_nan).std() * (252) ** (1/2)
      return roll_downside
    
def rolling_downside_risk_anualizado_hist(df, n):
    '''A função retorna o df de downside risk rolling anualizado de n dias com datas descasadas;
    df = dataframe utilizado para calcular downside risk rolling (apenas cotas);
    n = número de dias para o rolling
    EX: df_roll_vol = rolling_volatilidade(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    retornos_df = retorno_diario(df, 1)
    df_ret_negativo = retornos_df[retornos_df < 0]
    df_output = pd.DataFrame()
    for col in df_ret_negativo.columns:
        df_output[col] = df_ret_negativo[col].dropna().rolling(n).std() * (252) ** (1/2)
    return df_output

def tracking_error_rolling(df_cotas, df_benchmark, janela, benchmark, output_type = None):
    '''Retorna um dataframe de tracking error rolling;
    df_cotas = dataframe de cotas;
    df_benchmark = dataframe de cotas do benchmark;
    janela = janela do rolling;
    benchmark = string com o nome do fundo desejado;
    output_type = pode ser 0 (sem ser anualizado) ou 1 (anualizado)'''
    df = df_cotas.copy()
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, benchmark]

    # 2) Criando um dataframe com os dados do benchmark
    free_risk_columns = df.filter(like = benchmark).columns
    df_bench = pd.DataFrame()
    for i in free_risk_columns:
        df_bench[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno diário
    df_return = retorno_diario(df, 1)
    df_ret_bench = retorno_diario(df_bench, 1)

    # 4) Diferença entre fundos e benchmarks
    ## Renomeando as colunas do df_ret_bench
    for i, j in zip(df_ret_bench.columns, df.columns):
        df_ret_bench = df_ret_bench.rename(columns = {i: j})

    diff = df_return - df_ret_bench
    diff = (1 + df_return)/(1 + df_ret_bench) - 1

    # Desvio Padrão da diferença
    if output_type == None:
        tracking_error = rolling_volatilidade(diff, 'retorno', janela)
    elif output_type == 0:
        tracking_error = rolling_volatilidade(diff, 'retorno', janela)
    elif output_type == 1:
        tracking_error = rolling_volatilidade_anualizada(diff, 'retorno', janela)
    return tracking_error
    
#VaR 95%
def var(df, df_type, ic):
    '''A função retorna o df de volatilidade rolling de n dias;
    df = dataframe utilizado para calcular vol rolling (pode ser de cotas ou de retornos diários);
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    ic = intervalo de confiança para o VaR (em números inteiros: 99, 95, ...)
    EX: df_roll_vol = rolling_volatilidade_anualizada(df_retornos_diarios, df_type = 'retorno', n = 63)'''
    result = 100 - ic
    if df_type == 'retorno':
        VaR = pd.DataFrame()
        for fund in df.columns:
            VaR.loc[fund, 'VaR'] = np.percentile(df[fund], result)
        VaR = VaR.applymap(lambda x: "{:.2%}".format(x)) # formatando em porcentagem e com duas casas decimais
        return VaR
    elif df_type == 'cota':
        df_retorno = retorno_diario(df)
        VaR = pd.DataFrame()
        for fund in df_retorno.columns:
            VaR.loc[fund, 'VaR'] = np.percentile(df_retorno[fund], result)
        VaR = VaR.applymap(lambda x: "{:.2%}".format(x)) # formatando em porcentagem e com duas casas decimais
        return VaR
    
#Rolling VaR 95%
def roll_var(df, df_type, ic, n):
    '''Retorna o df de VaR rolling com nível de confiança a ser definido pelo usuário
    df = df utilizado para calcular var rolling;
    df_type = tipo de dados do df (pode ser 'cota' ou 'retorno');
    ic = intervalo de confiança do VaR (em número inteiro: 99, 95, ...
    n = número de dias para o rolling'''
    if df_type == 'retorno':
        result = (100 - ic)/100
        var_roll = df.rolling(n).quantile(result)
        return var_roll
    elif df_type == 'cota':
        result = (100 - ic)/100
        df_retornos = retorno_diario(df)
        var_roll = df_retornos.rolling(n).quantile(result)
        return var_roll
    
# Sharpe
def sharpe_rolling(df_cotas, df_benchmark, janela, free_risk):
    '''df_cotas =  df de cotas (deve conter uma coluna referente a taxa livre de risco: criar cotas fictícias para a taxa livre de risco);
    df_benchmark = df de cotas dos benchmarks que consta a taxa livre de risco;
    janela = janela de rolling desejada;
    free_risk = taxa livre de risco desejada (escrever como string: 'CDI')'''

    df = df_cotas.copy()
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like = free_risk).columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno Rolling da janela inputada em dias
    df_ret_janela = df / df.shift(janela) - 1 # df_ret_janela = (cota final) / (cota inicial) - 1
    df_ret_free_risk = df_free_risk / df_free_risk.shift(janela) - 1

    # 4) Retorno diário
    df_retornos = retorno_diario(df, method = 1)

    # 5) Rolling Volatilidade
    roll_vol = (df_retornos.rolling(janela).std()) * (252) ** (1/2)

    # 6) Cálculo do sharpe
    sharpe = pd.DataFrame()
    for col in df.columns:
        sharpe[col] = (df_ret_janela[col] - df_ret_free_risk[f'{col} - {free_risk}']) / roll_vol[col] 
    return sharpe

# Sharpe Histórico
def sharpe_historico(df, df_benchmark, free_risk):
    '''Retorna um dataframe com o sharpe diário histórico;
    df = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks;
    free_risk = string com o nome do benchmark desejado'''
    
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{free_risk}').columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno acumulado anualizado (ok)
    df_ret_acum =  retorno_acumulado_anualizado(df)
    df_ret_free_risk = retorno_acumulado_anualizado(df_free_risk)

    # 4) Volatilidade diária anualizada
    vol_diaria_anual = volatilidade_diaria_anualizada(df)

    # 5) Cálculo do sharpe
    sharpe = pd.DataFrame()
    for col in df.columns:
        sharpe[col] = (df_ret_acum[col] - df_ret_free_risk[f'{col} - {free_risk}']) / vol_diaria_anual[col] 
    return sharpe

# Sharpe Histórico
def sharpe_historico_nao_anualizado(df, df_benchmark, free_risk):
    '''Retorna um dataframe com o sharpe diário histórico;
    df = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks;
    free_risk = string com o nome do benchmark desejado'''
    
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{free_risk}').columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno acumulado anualizado (ok)
    df_ret_acum =  retorno_acumulado_cumprod(df, 0, input_type = 'with nan')
    df_ret_free_risk = retorno_acumulado_cumprod(df_free_risk, 0, 'with nan')

    # 4) Volatilidade diária anualizada
    vol_diaria_anual = volatilidade_diaria(df)

    # 5) Cálculo do sharpe
    sharpe = pd.DataFrame()
    for col in df.columns:
        sharpe[col] = (df_ret_acum[col] - df_ret_free_risk[f'{col} - {free_risk}']) / vol_diaria_anual[col] 
    return sharpe

# Sharpe Info
def sharpe_info(df_cotas, df_benchmark, free_risk):

    df = df_cotas.copy()

    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{free_risk}').columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno acumulado anualizado (ok)
    df_ret_acum =  retorno_acumulado_anualizado(df)
    df_ret_free_risk = retorno_acumulado_anualizado(df_free_risk)
    # Mudando os nomes das colunas
    df_ret_free_risk = mudar_nome_coluna(df_ret_free_risk, df.columns)

    # 5) Valores
    last_value_ret_acum = df_ret_acum.tail(1)
    last_value_ret_acum_free_risk = df_ret_free_risk.tail(1)

    sharpe_info = pd.DataFrame(columns = df.columns)
    for col in df.columns:
        sharpe_info.loc['Alpha', col] = last_value_ret_acum.iloc[0][col] - last_value_ret_acum_free_risk.iloc[0][col]
        sharpe_info.loc['Volatilidade Histórica', col] = volatilidade_anualizada(df[col])

    sharpe_info.loc['Sharpe'] = sharpe_info.loc['Alpha']/sharpe_info.loc['Volatilidade Histórica']

    return sharpe_info

# Sortino
def sortino_rolling(df, df_benchmark, janela, free_risk, min_valores_not_nan = None):
    '''A função retorna um dataframe de sortino rolling;
    df =  df de cotas (deve conter uma coluna referente a taxa livre de risco: criar cotas fictícias para a taxa livre de risco);
    df_benchmark = df de cotas dos benchmarks que consta a taxa livre de risco;
    janela = janela de rolling desejada;
    free_risk = taxa livre de risco desejada (escrever como string: 'CDI');
    min_valores_nao_nan = qntd mínimima de valores não nan para fazer o calculo da métrica
    '''
    # 1) Criando um dataframe com os dados do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    df_free_risk = pd.DataFrame()
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df_free_risk[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Retorno Rolling da janela inputada em dias
    df_ret_janela = df / df.shift(janela) - 1 # df_ret_janela = (cota final) / (cota inicial) - 1
    df_ret_free_risk = df_free_risk / df_free_risk.shift(janela) - 1 

    # 3) Retorno diário
    df_retornos = retorno_diario(df, method = 1)
    df_retornos_negativos = df_retornos[df_retornos < 0]


    # 4) Rolling Volatilidade
    roll_vol = (df_retornos_negativos.rolling(janela, min_periods = min_valores_not_nan).std()) * (252) ** (1/2)

    # 5) Cálculo do sortino
    sortino = pd.DataFrame()
    alfa = pd.DataFrame()
    for col in df.columns:
        sortino[col] = (df_ret_janela[col] - df_ret_free_risk[f'{col} - {free_risk}']) / roll_vol[col]
    return sortino

# Sortino Histórico
def sortino_historico(df, df_benchmark, free_risk):
    '''Retorna um dataframe com o sortino diário histórico;
    df = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks;
    free_risk = string com o nome do benchmark desejado'''
    
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{free_risk}').columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, 1)

    # 3) Retorno acumulado anualizado (ok)
    df_ret_acum =  retorno_acumulado_anualizado(df)
    df_ret_free_risk = retorno_acumulado_anualizado(df_free_risk)

    # 4) Volatilidade diária anualizada
    downside_vol_anual = downside_risk_anualizado(df)

    # 5) Calculo do sortino
    sortino = pd.DataFrame()
    alfa = pd.DataFrame()
    for col in df.columns:
        sortino[col] = (df_ret_acum[col] - df_ret_free_risk[f'{col} - {free_risk}']) / downside_vol_anual[col]
    return sortino

# Sortino Info
def sortino_info(df_cotas, df_benchmark, free_risk):

    df = df_cotas.copy()

    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {free_risk}'] = df_benchmark.loc[data:, free_risk]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{free_risk}').columns
    df_free_risk = pd.DataFrame()
    for i in free_risk_columns:
        df_free_risk[i] = df[i]
        df = df.drop(i, axis = 1)

    # 3) Retorno acumulado anualizado (ok)
    df_ret_acum =  retorno_acumulado_anualizado(df)
    df_ret_free_risk = retorno_acumulado_anualizado(df_free_risk)
    # Mudando os nomes das colunas
    df_ret_free_risk = mudar_nome_coluna(df_ret_free_risk, df.columns)

    # 5) Valores
    last_value_ret_acum = df_ret_acum.tail(1)
    last_value_ret_acum_free_risk = df_ret_free_risk.tail(1)

    sortino_info = pd.DataFrame(columns = df.columns)
    for col in df.columns:
        sortino_info.loc['Alpha', col] = last_value_ret_acum.iloc[0][col] - last_value_ret_acum_free_risk.iloc[0][col]
        sortino_info.loc['Downside Risk Histórico', col] = downside_risk_anualizado(df[col])

    sortino_info.loc['Sortino'] = sortino_info.loc['Alpha']/sortino_info.loc['Downside Risk Histórico']

    return sortino_info

# Information Ratio
def information_ratio_rolling(df_cotas, df_benchmark, janela, benchmark):
    df = df_cotas.copy()
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, {benchmark}]

    # 2) Criando um dataframe com os dados do free risk rate
    free_risk_columns = df.filter(like=f'{benchmark}').columns
    df_bench = pd.DataFrame()
    for i in free_risk_columns:
        df_bench[i] = df[i]
        df = df.drop(i, 1)

    # 3) Retorno Rolling da janela inputada em dias
    df_ret_janela = df / df.shift(janela) - 1 # df_ret_janela = (cota final) / (cota inicial) - 1
    df_ret_bench = df_bench / df_bench.shift(janela) - 1

    # 4) Diferença entre fundos e benchmarks
    ## Renomeando as colunas do df_ret_bench
    for i, j in zip(df_ret_bench.columns, df.columns):
        df_ret_bench = df_ret_bench.rename(columns = {i: j})

    excesso_de_retorno = (1 + df_ret_janela) / (1 + df_ret_bench) - 1
    # 5) Tracking Error
    te_roll = tracking_error_rolling(df, df_benchmark, janela, benchmark)

    # 6) Information Ratio
    information_ratio = excesso_de_retorno / te_roll

    return information_ratio

#Calmar
def calmar_rolling(df, free_risk):
    '''df =  df de cotas (deve conter uma coluna referente a taxa livre de risco: criar cotas fictícias para a taxa livre de risco);
    free_risk = taxa livre de risco desejada (escrever como string: 'CDI')'''

    # 1) Calculando os retornos necessários
    df_retornos = retorno_diario(df)
    df_ret_acum = retorno_acumulado(df)

    # 3) Rolling de MDD
    ret_acum_plus1 = df_ret_acum + 1
    max_1 = ret_acum_plus1.cummax()
    drawdown = ret_acum_plus1 / max_1 - 1

    # 2) Anualizando os retornos acumulados inputados e o MDD:
    df_acum_anual = ((1 + df_ret_acum) ** (252/len(df_ret_acum)) - 1)
    drawdown_anual = ((1 + drawdown) ** (252/len(drawdown)) - 1)

    # 4) Calculando o calmar rolling
    df_calmar = pd.DataFrame()
    for col in df_acum_anual.columns:
        for index in df_acum_anual.index:
            if drawdown_anual.loc[index, col] == 0:
                df_calmar.loc[index, col] = 0   
            else:
                df_calmar.loc[index, col] = (df_acum_anual[index, col] - df_acum_anual.loc[index, free_risk])/drawdown_anual.loc[index, col]
    
    df_calmar = df_calmar.drop(free_risk , axis = 1)
    return df_calmar

#Análise De Desempenho
def analise_de_desempenho(df_ativo, df_bench, fundo, benchmark):
    '''df_ativo = df de cotas dos fundos/ativos [sem tratamento para os NaNs]);
    df_bench = df de cotas dos benchmarks;
    fundo = string do nome do fundo conforme escrito na coluna do df_ativo;
    benchmark = string do nome do benchmark conforme escrito na coluna do df_ativo'''

    # 1) Criando um dataframe novo com as cotas do fundo foco e com as cotas do CDI casadas com as datas do fundo foco
    df = pd.DataFrame()
    df[fundo] = df_ativo[fundo]
    df[benchmark] = df_bench[benchmark]

    # 2) Separando o dataframe de cotas por ano e pegando algumas informações que serão posteriormente úteis
    year_group = df.groupby(df.index.year)

    data_inicial = df.index[0] # Primeira data do dataframe
    first_year = df.index[0].year # Primeiro ano do dataframe
    last_year = df.index[-1].year # Ano mais recente do dataframe

    # 3) Criando uma lista com os anos que foram analisados e um dicionário salvando as cotas referentes à cada ano OK
    anos = []

    dict = {}

    for i in range(first_year, last_year + 1):
        anos.append(i)
        dict[f'cotas_{i}'] = year_group.get_group(i)

    print(f'dataframes separados com dados dos seguintes anos: {anos} (exemplo: cotas_{i})')

    # 4) Criando um dataframe que será preenchido no loop em seguida
    df_desempenho = pd.DataFrame()

    for ano in range(first_year, last_year + 1):
        # Para o primeiro ano, o cálculo da rentabilidade é feito com base na primeira cota disponível do fundo contra a última cota do fundo
        if ano == first_year:
            df_aux = dict[f'cotas_{ano}']

            # Retorno diário de cada ano
            dict[f'retornos_{ano}'] = retorno_diario(df_aux)
            df_aux_retornos = dict[f'retornos_{ano}']
            
            # Retorno Acumulado de cada ano:
            dict[f'ret_acum_{ano}'] = retorno_acumulado(df_aux)
            df_aux_ret_acum = dict[f'ret_acum_{ano}']
            dict[f'valor_retorno_{ano}'] = df_aux_ret_acum.iloc[-1]
            df_aux_valor_retorno = dict[f'valor_retorno_{ano}']
            df_desempenho.loc[ano, 'Rentabilidade'] = df_aux_valor_retorno[fundo]
            df_desempenho.loc[ano, benchmark] = df_aux_valor_retorno[benchmark]

            # Volatilidade anualizada:
            dict[f'vol_anualizada_{ano}'] = df_aux_retornos.std() * (252) ** (1/2)
            df_aux_vol_anualizada = dict[f'vol_anualizada_{ano}']
            df_desempenho.loc[ano, 'Volatilidade Anualizada'] = df_aux_vol_anualizada[fundo]

    ##################
        else:
            # Para os demais anos o cálculo da rentabilidade é feito considerando a última cota do ano (N - 1) com a última cota do ano N
            ano_anterior = ano - 1
            # dataframe do ano anterior
            df_ano_anterior = dict[f'cotas_{ano_anterior}']
            # Última data com cota do ano anterior
            last_index_df_ano_anterior = df_ano_anterior.index[-1]
            # Dataframe com as cotas do ano atual
            df_aux = dict[f'cotas_{ano}']
            # Adicionando a última cota do ano anterior no dataframe do ano atual
            df_aux = pd.concat([df_ano_anterior.loc[[last_index_df_ano_anterior]], df_aux])

            # O restante é igual à parte do 'if' acima

            # Retorno diário de cada ano
            dict[f'retornos_{ano}'] = retorno_diario(df_aux)
            df_aux_retornos = dict[f'retornos_{ano}']
            
            # Retorno Acumulado de cada ano:
            dict[f'ret_acum_{ano}'] = retorno_acumulado(df_aux)
            df_aux_ret_acum = dict[f'ret_acum_{ano}']
            dict[f'valor_retorno_{ano}'] = df_aux_ret_acum.iloc[-1]
            df_aux_valor_retorno = dict[f'valor_retorno_{ano}']
            df_desempenho.loc[ano, 'Rentabilidade'] = df_aux_valor_retorno[fundo]
            df_desempenho.loc[ano, benchmark] = df_aux_valor_retorno[benchmark]

            # Volatilidade anualizada:
            dict[f'vol_anualizada_{ano}'] = df_aux_retornos.std() * (252) ** (1/2)
            df_aux_vol_anualizada = dict[f'vol_anualizada_{ano}']
            df_desempenho.loc[ano, 'Volatilidade Anualizada'] = df_aux_vol_anualizada[fundo]

    df_desempenho.loc['Média'] = df_desempenho.mean()
    df_desempenho.loc['Máx'] = df_desempenho.max()
    df_desempenho.loc['Min'] = df_desempenho.min()
    print(f'Análise de Desempenho do fundo {fundo}')
    print(f'Data de início do fundo: {data_inicial}')
    #df_desempenho = df_desempenho.applymap(lambda x: "{:.2%}".format(x))
    return df_desempenho

# Análise de Desempenho Rolling
def analise_de_desempenho_rolling(df_ativo, df_bench, fundo, benchmark, fof = None):
    '''df_ativo = df de cotas dos fundos/ativos;
    df_bench = df de cotas dos benchmarks;
    fundo = string do nome do fundo conforme escrito na coluna do df_ativo;
    benchmark = string do nome do benchmark conforme escrito na coluna do df_ativo;
    fof = string com o nome da caixa do fof que análise está sendo feita (possibilidades: 'MM', 'FIA', 'FIE', '4994')'''

    # 1) Pegando as datas de início e fim do fundo foco
    primeiras_datas_validas = df_ativo.apply(pd.Series.first_valid_index)
    ultimas_datas_validas = df_ativo.apply(pd.Series.last_valid_index)
    data_inicio = primeiras_datas_validas[fundo]
    data_fim = ultimas_datas_validas[fundo]
    # 2) Vendo a diferença entre fim e início em meses
    dias = data_fim - data_inicio
    dias = dias.days
    meses = dias/21
    meses = int(meses) # qntd de meses de existência do fundo transformado em um número inteiro
    resto = meses % 6 # 3) Dividindo o a qntd de meses por 6 e pegando o resto
    n_mais_proximo = meses - resto # Número divisível por 6 mais próximo do número de meses
    # 3) Criando uma lista
    lista_aux = []
    for i in range(6, n_mais_proximo + 1, 6):
            lista_aux.append(i) # Preenchendo a lista_aux

    lista = lista_aux[:4]
    for i in range(4, len(lista_aux)):
        avaliaçao = i % 2
        valor = lista_aux[i]
        if avaliaçao != 0:
            lista.append(valor)
        else:
            pass

    # 4) Criando dfs para cada período definido na lista
    ## 4.1) Consolidando cotas do fundo foco com o benchmark 
    df = pd.DataFrame()
    df[fundo] = df_ativo[fundo]
    df[benchmark] = df_bench[benchmark]
    dict_df = {}

    ## 4.2) Salvando os dfs referentes a cada período em um dicionário
    for periodo in lista:
        valor = periodo * 21 # exemplo: valor = 6 * 21 (6 meses); valor = 12 * 21 (12 meses)
        dict_df[f'{periodo}m'] = df.tail(valor)

    # 5) Criando o df de output
    df_desempenho = pd.DataFrame()

    for periodo in lista:
        df_aux = dict_df[f'{periodo}m']
        # Retorno Acumulado
        df_ret_acum = retorno_acumulado(df_aux)
        valor_retorno = df_ret_acum.iloc[-1]
        # Volatilidade
        vol = volatilidade_anualizada(df_aux)
        # Preenchendo o df_desempenho
        df_desempenho.loc[f'{periodo}m', fundo] = valor_retorno[fundo]
        df_desempenho.loc[f'{periodo}m', benchmark] = valor_retorno[benchmark]
        df_desempenho.loc[f'{periodo}m', f'Volatilidade Anualizada {fundo}'] = vol[fundo]
        if fof != None and fof != 'MM':
            df_desempenho.loc[f'{periodo}m', f'Volatilidade Anualizada {benchmark}'] = vol[benchmark]

    df_desempenho.loc['Média'] = df_desempenho.mean()
    df_desempenho.loc['Máx'] = df_desempenho.max()
    df_desempenho.loc['Min'] = df_desempenho.min()
    print(f'Análise de Desempenho rolling do fundo {fundo}')
    #df_desempenho = df_desempenho.applymap(lambda x: "{:.2%}".format(x))

    return df_desempenho


#Evolução de Quartil
def evoluçao_de_quartil_anual(df_cotas):
    '''A função retorna um dataframe com notas de quartil com base 
    na performance comparativa entre os fundos;
    df_cotas = df de cotas/preços'''
    performance = performance_anualizada_por_ano(df_cotas, output_type = 0) # ok
    df = performance.copy()
    quartis = pd.DataFrame()
    for i in df.index:
        quartis.loc[i, 'q1'] = df.loc[i].dropna().quantile(0.25)
        quartis.loc[i, 'q2'] = df.loc[i].dropna().quantile(0.5)
        quartis.loc[i, 'q3'] = df.loc[i].dropna().quantile(0.75)
        quartis.loc[i, 'q4'] = df.loc[i].dropna().quantile(1.00)
    print(quartis)
    
    df2 = pd.DataFrame()
    for j in performance.columns:
        for i in performance.index:
            valor = performance.loc[i, j]
            q1 = quartis.loc[i, 'q1']
            q2 = quartis.loc[i, 'q2']
            q3 = quartis.loc[i, 'q3']
            q4 = quartis.loc[i, 'q4']
            if valor <= q1:
                df2.loc[i, j] = 1
            elif valor <= q2 and valor > q1:
                df2.loc[i, j] = 2
            elif valor <= q3 and valor > q2:
                df2.loc[i, j] = 3
            elif valor <= q4 and valor > q3:
                df2.loc[i, j] = 4
            elif valor > q4:
                df2.loc[i, j] = 5
            elif math.isnan(valor):
                df2.loc[i, j] = np.nan
    
    return df2

#Evolução de Quartil Mensal
def evoluçao_de_quartil_mensal(df_cotas):
    '''A função retorna um dataframe com notas de quartil com base 
    na performance comparativa entre os fundos;
    df_cotas = df de cotas/preços'''
    df_retornos = retorno_diario(df_cotas)
    performance = performance_por_mes(df_retornos, 0)
    df = performance.copy()
    quartis = pd.DataFrame()
    for i in df.index:
        quartis.loc[i, 'q1'] = df.loc[i].dropna().quantile(0.25)
        quartis.loc[i, 'q2'] = df.loc[i].dropna().quantile(0.5)
        quartis.loc[i, 'q3'] = df.loc[i].dropna().quantile(0.75)
        quartis.loc[i, 'q4'] = df.loc[i].dropna().quantile(1.00)
    print(quartis)
    
    df2 = pd.DataFrame()
    for j in performance.columns:
        for i in performance.index:
            valor = performance.loc[i, j]
            q1 = quartis.loc[i, 'q1']
            q2 = quartis.loc[i, 'q2']
            q3 = quartis.loc[i, 'q3']
            q4 = quartis.loc[i, 'q4']
            if valor <= q1:
                df2.loc[i, j] = 1
            elif valor <= q2 and valor > q1:
                df2.loc[i, j] = 2
            elif valor <= q3 and valor > q2:
                df2.loc[i, j] = 3
            elif valor <= q4 and valor > q3:
                df2.loc[i, j] = 4
            elif valor > q4:
                df2.loc[i, j] = 4
            elif math.isnan(valor):
                df2.loc[i, j] = np.nan
    
    return df2

#Proporção Quartil
def proporçao_quartil(df):
    '''A função retorna um dataframe com a proporção de vezes que o fundo encontrou-se 
    no primeiro, segundo, terceiro e quarto quartil;
    df: dataframe de evoluçao de quartil gerado pela função evoluçao_de_quartil_mensal 
    ou evoluçao_de_quartil_anual'''
    dict_qntd_q1 = {}
    dict_qntd_q2 = {}
    dict_qntd_q3 = {}
    dict_qntd_q4 = {}

    for col in df.columns:
        value = df[col].value_counts()
        dict_qntd_q1[col] = value.get(1, 0)
        dict_qntd_q2[col] = value.get(2, 0)
        dict_qntd_q3[col] = value.get(3, 0)
        dict_qntd_q4[col] = value.get(4, 0)

    #total = len(df.index)
    proporçao = pd.DataFrame(index = ['% q1', '% q2', '% q3', '% q4'])
    for col in df.columns:
        sample = df[col].dropna()
        total = len(sample)
        valor_q1 = dict_qntd_q1[col]
        valor_q2 = dict_qntd_q2[col]
        valor_q3 = dict_qntd_q3[col]
        valor_q4 = dict_qntd_q4[col]

        # Proporção da nota do quartil
        prop_q1 = valor_q1/total
        prop_q2 = valor_q2/total
        prop_q3 = valor_q3/total
        prop_q4 = valor_q4/total

        # Preenchendo o df proporçao
        proporçao.loc['% q1', col] = prop_q1
        proporçao.loc['% q2', col] = prop_q2
        proporçao.loc['% q3', col] = prop_q3
        proporçao.loc['% q4', col] = prop_q4
    
    return proporçao

#Quartil Index (é possível melhorar a função iterando no index tbm)
def quartil_index(df_proporçao):
    '''df_proporçao: df retornado pela função proporçao_quartil;'''
    dict_index_quartil = {}
    for col in df_proporçao.columns:
        valor_q1 = df_proporçao.loc['% q1', col]
        valor_q2 = df_proporçao.loc['% q2', col]
        valor_q3 = df_proporçao.loc['% q3', col]
        valor_q4 = df_proporçao.loc['% q4', col]

        #index = (1 + (valor_q3 + valor_q4))/(1 + (valor_q1 + valor_q2)) - 1

        index_aux = (valor_q1 + 2 * valor_q2 + 3 * valor_q3 + 4 * valor_q4)/10
        index = (index_aux/0.4) * 4

        # Preenchendo o dict index_quartil:
        dict_index_quartil[col] = index
    index_quartil = pd.DataFrame.from_dict(dict_index_quartil, orient = 'index')
    index_quartil = index_quartil.sort_values(by = 0, ascending = False)
    index_quartil.loc['Média'] = index_quartil.mean()
    index_quartil.loc['Máx'] = index_quartil.max()
    index_quartil.loc['Min'] = index_quartil.min()
    return index_quartil

#Comparativo Volatilidade Histórico
def comparativo_vol_historica(df_ativo):
    '''A função retorna um dataframe de comparativo histórico de volatilidade;
    df_ativo = df de cotas desejado'''
    df_retornos = retorno_diario(df_ativo, 1)
    comparativo = pd.DataFrame(columns = df_ativo.columns)
    for i in df_retornos.columns:
        comparativo.loc['12m', i] = df_retornos[i].tail(252).std() * (252) ** (1/2)
        comparativo.loc['24m', i] = df_retornos[i].tail(504).std() * (252) ** (1/2)
        comparativo.loc['Histórica', i] = df_retornos[i].std() * (252) ** (1/2)
    
    #comparativo = comparativo.applymap(lambda x: "{:.2%}".format(x))
    return comparativo.T

# Comparativo Downside risk histórico
def comparativo_downside_risk_historico(df_ativo):
    '''A função retorna um dataframe de comparativo histórico de downside risk;
    df_ativo = df de cotas desejado'''
    df_retornos = retorno_diario(df_ativo, 1)
    df_ret_negativos = df_retornos[df_retornos < 0]
    comparativo = pd.DataFrame(columns = df_ativo.columns)
    for i in df_ret_negativos.columns:
        comparativo.loc['12m', i] = df_ret_negativos[i].tail(252).std() * (252) ** (1/2)
        comparativo.loc['24m', i] = df_ret_negativos[i].tail(504).std() * (252) ** (1/2)
        comparativo.loc['Histórica', i] = df_ret_negativos[i].std() * (252) ** (1/2)

    #comparativo = comparativo.applymap(lambda x: "{:.2%}".format(x))
    return comparativo.T

#Comparativo Sharpe
def comparativo_sharpe(df_cotas, df_benchmark, benchmark):
    '''Retorna um dataframe de Comparativo Histórico de Sharpe;
    df_cotas = dataframe de cotas;
    df_benchmark = daraframe de cotas dos benchmarks;
    benchmark = string definindo o benchmark desejado'''

    # Criando dfs
    df_12m = df_cotas.tail(252)
    df_12m[benchmark] = df_benchmark[benchmark]

    df_24m = df_cotas.tail(504)
    df_24m[benchmark] = df_benchmark[benchmark]

    df_historico = df_cotas.copy()
    primeiras_datas_validas = df_historico.apply(pd.Series.first_valid_index)
    for i in df_historico.columns:
        data = primeiras_datas_validas[i]
        df_historico[f'{benchmark} - {i}'] = df_benchmark.loc[data:, benchmark]


    # Armazenando dfs em um dict
    dict = {'12m': df_12m,
            '24m': df_24m,
            'historico': df_historico}
    lista = ['12m', '24m', 'historico']
    new_dict = {}
    
    # Retorno acumulado anualizado
    for i in lista:
        if i != 'historico':
            df = dict[i]
            df_ret_acum_anualizado = retorno_acumulado_anualizado(df)
            valor_retorno_bench = df_ret_acum_anualizado.iloc[-1, -1]
            df_ret_acum_anualizado = df_ret_acum_anualizado.drop(benchmark, axis = 1)
            df = df.drop(benchmark, axis = 1)
            # Salvando alterações e novos dfs no dict
            new_dict[f'valor_retorno_bench_{i}'] = valor_retorno_bench
            new_dict[f'ret_acum_anualizado_{i}'] = df_ret_acum_anualizado
            new_dict[i] = df
        else:
            df = dict[i]
            df_ret_acum_anualizado = retorno_acumulado_anualizado(df)
            bench_columns = df_ret_acum_anualizado.filter(like=benchmark).columns
            for col in bench_columns:
                new_dict[col] = df_ret_acum_anualizado[col].iloc[-1]
                df_ret_acum_anualizado = df_ret_acum_anualizado.drop(col, axis = 1)
                df = df.drop(col, axis = 1)
            new_dict[f'ret_acum_anualizado_{i}'] = df_ret_acum_anualizado
            new_dict[i] = df
    df_comparativo = pd.DataFrame()

    for i in lista:
        if i != 'historico':
            # definindo os dfs do dict que serão utilizados
            df = new_dict[i]
            df_ret_acum_anualizado = new_dict[f'ret_acum_anualizado_{i}']
            valor_retorno_bench = new_dict[f'valor_retorno_bench_{i}']
            # calculando novas métricas
            serie_vol_anualizada = volatilidade_anualizada(df)
            df_vol_anualizada = serie_vol_anualizada.to_frame()
            new_dict[f'df_vol_anualizada_{i}'] = df_vol_anualizada
            # Pegando dados necessários e salvando no dict
            serie_valor_retorno = df_ret_acum_anualizado.iloc[-1] #pegando a ultima linha de ret_acum do df de ret_acum
            df_valor_retorno = serie_valor_retorno.to_frame() # convertendo essa linha em um dataframe
            new_dict[f'df_valor_retorno_{i}'] = df_valor_retorno # salvando essa ultima linha do dict
            # Pegando mais alguns dados relevantes para efetuar cálculos
            col = df_valor_retorno.columns[0] # O nome da primeira coluna é 
            # Preenchendo o df_comparativo
            for j in df_valor_retorno.index:
                df_comparativo.loc[i, j] = (df_valor_retorno.loc[j, col] - valor_retorno_bench)/df_vol_anualizada.loc[j, 0]
        else:
            # definindo os dfs do dict que serão utilizados
            df = new_dict[i]
            df_ret_acum_anualizado = new_dict[f'ret_acum_anualizado_{i}']
            # calculando novas métricas
            serie_vol_anualizada = volatilidade_anualizada(df)
            df_vol_anualizada = serie_vol_anualizada.to_frame()
            new_dict[f'df_vol_anualizada_{i}'] = df_vol_anualizada
            # Pegando dados necessários e salvando no dict
            serie_valor_retorno = df_ret_acum_anualizado.iloc[-1] #pegando a ultima linha de ret_acum do df de ret_acum
            df_valor_retorno = serie_valor_retorno.to_frame() # convertendo essa linha em um dataframe
            new_dict[f'df_valor_retorno_{i}'] = df_valor_retorno # salvando essa ultima linha do dict
            # Pegando mais alguns dados relevantes para efetuar cálculos
            col = df_valor_retorno.columns[0] # O nome da primeira coluna é 
            for coluna in df.columns:
                valor_retorno_bench = new_dict[f'{benchmark} - {coluna}']
                # Preenchendo o df_comparativo
                df_comparativo.loc[i, coluna] = (df_valor_retorno.loc[coluna, col] - valor_retorno_bench)/df_vol_anualizada.loc[coluna, 0]
        
    return df_comparativo.T

# Comparativo Sortino
def comparativo_sortino(df_cotas, df_benchmark, benchmark):
    '''Retorna um dataframe de Comparativo Histórico de Sharpe;
    df_cotas = dataframe de cotas;
    df_benchmark = daraframe de cotas dos benchmarks;
    benchmark = string definindo o benchmark desejado'''

    # Criando dfs
    df_12m = df_cotas.tail(252)
    df_12m[benchmark] = df_benchmark[benchmark]

    df_24m = df_cotas.tail(504)
    df_24m[benchmark] = df_benchmark[benchmark]

    df_historico = df_cotas.copy()
    primeiras_datas_validas = df_historico.apply(pd.Series.first_valid_index)
    for i in df_historico.columns:
        data = primeiras_datas_validas[i]
        df_historico[f'{benchmark} - {i}'] = df_benchmark.loc[data:, benchmark]


    # Armazenando dfs em um dict
    dict = {'12m': df_12m,
            '24m': df_24m,
            'historico': df_historico}
    lista = ['12m', '24m', 'historico']
    new_dict = {}
    
    # Retorno acumulado anualizado
    for i in lista:
        if i != 'historico':
            df = dict[i]
            df_ret_acum_anualizado = retorno_acumulado_anualizado(df)
            valor_retorno_bench = df_ret_acum_anualizado.iloc[-1, -1]
            df_ret_acum_anualizado = df_ret_acum_anualizado.drop(benchmark, axis = 1)
            df = df.drop(benchmark, axis = 1)
            # Salvando alterações e novos dfs no dict
            new_dict[f'valor_retorno_bench_{i}'] = valor_retorno_bench
            new_dict[f'ret_acum_anualizado_{i}'] = df_ret_acum_anualizado
            new_dict[i] = df
        else:
            df = dict[i]
            df_ret_acum_anualizado = retorno_acumulado_anualizado(df)
            bench_columns = df_ret_acum_anualizado.filter(like=benchmark).columns
            for col in bench_columns:
                new_dict[col] = df_ret_acum_anualizado[col].iloc[-1]
                df_ret_acum_anualizado = df_ret_acum_anualizado.drop(col, axis = 1)
                df = df.drop(col, axis = 1)
            new_dict[f'ret_acum_anualizado_{i}'] = df_ret_acum_anualizado
            new_dict[i] = df
            
    df_comparativo = pd.DataFrame()

    for i in lista:
        if i != 'historico':
            # definindo os dfs do dict que serão utilizados
            df = new_dict[i]
            df_ret_acum_anualizado = new_dict[f'ret_acum_anualizado_{i}']
            valor_retorno_bench = new_dict[f'valor_retorno_bench_{i}']
            # Calculando downside risk
            serie_vol_anualizada = downside_risk_anualizado(df)
            df_vol_anualizada = serie_vol_anualizada.to_frame()
            new_dict[f'df_vol_anualizada_{i}'] = df_vol_anualizada
            # Pegando dados necessários e salvando no dict
            serie_valor_retorno = df_ret_acum_anualizado.iloc[-1] #pegando a ultima linha de ret_acum do df de ret_acum
            df_valor_retorno = serie_valor_retorno.to_frame() # convertendo essa linha em um dataframe
            new_dict[f'df_valor_retorno_{i}'] = df_valor_retorno # salvando essa ultima linha do dict
            # Pegando mais alguns dados relevantes para efetuar cálculos
            col = df_valor_retorno.columns[0] # O nome da primeira coluna é 
            # Preenchendo o df_comparativo
            for j in df_valor_retorno.index:
                df_comparativo.loc[i, j] = (df_valor_retorno.loc[j, col] - valor_retorno_bench)/df_vol_anualizada.loc[j, 0]
        else:
            # definindo os dfs do dict que serão utilizados
            df = new_dict[i]
            df_ret_acum_anualizado = new_dict[f'ret_acum_anualizado_{i}']
            # calculando novas métricas
            serie_vol_anualizada = downside_risk_anualizado(df)
            df_vol_anualizada = serie_vol_anualizada.to_frame()
            new_dict[f'df_vol_anualizada_{i}'] = df_vol_anualizada
            # Pegando dados necessários e salvando no dict
            serie_valor_retorno = df_ret_acum_anualizado.iloc[-1] #pegando a ultima linha de ret_acum do df de ret_acum
            df_valor_retorno = serie_valor_retorno.to_frame() # convertendo essa linha em um dataframe
            new_dict[f'df_valor_retorno_{i}'] = df_valor_retorno # salvando essa ultima linha do dict
            # Pegando mais alguns dados relevantes para efetuar cálculos
            col = df_valor_retorno.columns[0] # O nome da primeira coluna é 
            for coluna in df.columns:
                valor_retorno_bench = new_dict[f'{benchmark} - {coluna}']
                # Preenchendo o df_comparativo
                df_comparativo.loc[i, coluna] = (df_valor_retorno.loc[coluna, col] - valor_retorno_bench)/df_vol_anualizada.loc[coluna, 0]
        
    return df_comparativo.T


#Alpha
        
def alpha(df_cotas, df_benchmark, benchmark, method = None):
    '''A função retorna um dataframe com os alphas calculados;
    df_cotas = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks (até mesmo o CDI precisa estar em formato de cota);
    benchmark = string com o nome do benchmark de referência que se deseja calcular o alpha;
    method = método de calculo do alpha (inputs possíveis: 'retorno diario', 'retorno acumulado', 'retornos acumulado anualizado').
    #Ex: caso method = 'retorno diario' o alpha é calculado com base no retorno diário dos ativos;
    #Ex2: caso method = 'retorno acumulado anualizado', o alpha é calculado com base no retorno acumulado anualizado dos ativos
    '''
    if method == 'retorno diario':
        df = df_cotas.copy()
        # add coluna do benchmark
        primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
        for col in df.columns:
            data = primeiras_datas_validas[col]
            df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, benchmark]
        ################################################################################################################
        df_retornos = retorno_diario(df, 1)
        ################################################################################################################
        alpha = df_retornos.copy()
        alpha = alpha + 1
        # calculo do alpha 
        for fundo in df_cotas.columns:
            alpha[f'alpha - {fundo} x {benchmark}'] = (alpha[f'{fundo}']/alpha[f'{fundo} - {benchmark}']) - 1
        # criando um novo df que só tem os valores do alpha calculado
        alpha_columns = alpha.filter(like='alpha').columns
        alpha_final = pd.DataFrame()
        for col in alpha_columns:
            alpha_final[col] = alpha[col]

        for col_nova, col_antiga in zip(df_cotas.columns, alpha_final.columns):
            alpha_final = alpha_final.rename(columns = {col_antiga: col_nova})
        return alpha_final
    
    if method == 'retorno acumulado':
        df = df_cotas.copy()
        # add coluna do benchmark
        primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
        for col in df.columns:
            data = primeiras_datas_validas[col]
            df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, benchmark]
        ################################################################################################################
        df_retornos = retorno_acumulado_cumprod(df, 0, input_type = 'with nan')
        ################################################################################################################
        alpha = df_retornos.copy()
        alpha = alpha + 1
        # calculo do alpha 
        for fundo in df_cotas.columns:
            alpha[f'alpha - {fundo} x {benchmark}'] = (alpha[f'{fundo}']/alpha[f'{fundo} - {benchmark}']) - 1
        # criando um novo df que só tem os valores do alpha calculado
        alpha_columns = alpha.filter(like='alpha').columns
        alpha_final = pd.DataFrame()
        for col in alpha_columns:
            alpha_final[col] = alpha[col]

        for col_nova, col_antiga in zip(df_cotas.columns, alpha_final.columns):
            alpha_final = alpha_final.rename(columns = {col_antiga: col_nova})
        return alpha_final
    
    if method == 'retorno acumulado anualizado':
        df = df_cotas.copy()
        # add coluna do benchmark
        primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
        for col in df.columns:
            data = primeiras_datas_validas[col]
            df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, benchmark]
        ################################################################################################################
        df_retornos_aux = retorno_acumulado_cumprod(df, 0, input_type = 'with nan')
        df_retornos = anualizar_taxa(df_retornos_aux)
        ################################################################################################################
        alpha = df_retornos.copy()
        alpha = alpha + 1
        # calculo do alpha 
        for fundo in df_cotas.columns:
            alpha[f'alpha - {fundo} x {benchmark}'] = (alpha[f'{fundo}']/alpha[f'{fundo} - {benchmark}']) - 1
        # criando um novo df que só tem os valores do alpha calculado
        alpha_columns = alpha.filter(like='alpha').columns
        alpha_final = pd.DataFrame()
        for col in alpha_columns:
            alpha_final[col] = alpha[col]

        for col_nova, col_antiga in zip(df_cotas.columns, alpha_final.columns):
            alpha_final = alpha_final.rename(columns = {col_antiga: col_nova})
        return alpha_final

#Alpha Rolling
def alpha_rolling(df_cotas, df_benchmark, benchmark, janela):
    '''A função retorna um df de alfa rolling;
    df_cotas = dataframe com as cotas do fundo e do bench;
    df_benchmark = dataframe com as cotas dos benchmarks;
    focus = string com o nome do fundo foco da DD;
    benchmark = string com o nome do benchmark desejado;
    janela = janela do rolling;'''

    df = df_cotas.copy()
    # 1) add coluna do benchmark
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {benchmark}'] = df_benchmark.loc[data:, benchmark]

    ##############
    # 2) Retorno Rolling da janela inputada em dias
    df_ret_janela = df / df.shift(janela) - 1
    ##############

    # 3) Cálculo do Alpha
    alpha = df_ret_janela.copy()
    alpha = alpha + 1
    for fundo in df_cotas.columns:
        alpha[f'alpha - {fundo} x {benchmark}'] = (alpha[f'{fundo}']/alpha[f'{fundo} - {benchmark}']) - 1

    # 4) criando um novo df que só tem os valores do alpha calculado
    alpha_columns = alpha.filter(like='alpha').columns
    alpha_final = pd.DataFrame()
    for col in alpha_columns:
        alpha_final[col] = alpha[col]

    #alpha_final.replace([np.inf, -np.inf], 0, inplace = True)
    return alpha_final

#Comparativo Alpha
def comparativo_alpha(df_cotas, df_benchmark, benchmark, method):
    '''A função retorna uma tabela com o alfa 12m, 24m e Histórico de cada um dos ativos presentes em df_cotas;
    df_cotas = dataframe de cotas dos ativos/fundos;
    df_benchmark = dataframe com as cotas do benchmark (OBS: CDI precisa estar em formato de cota);
    benchmark = benchmark de referência para calcular o alpha;
    method = método de cálculo do alpha (inputs possíveis: 'retorno diario', 'retorno acumulado', 'retornos acumulado anualizado').
    #Ex: caso method = 'retorno diario', o alpha é calculado com base no retorno diário dos ativos;
    #Ex2: caso method = 'retorno acumulado anualizado', o alpha é calculado com base no retorno acumulado anualizado dos ativos
    '''
    df = df_cotas.copy()
    # Definindo os dfs de 12 meses de cotas e de 24 meses de cotas e calculando o retorno acumulado de cada um deles
    df_12_meses = df.tail(252)
    df_24_meses = df.tail(2 * 252)
    ## OBS: O df histórico é o próprio df de input
    # Calculando os alphas
    alfa_12_meses = alpha(df_12_meses, df_benchmark, benchmark, method = method)
    alfa_24_meses = alpha(df_24_meses, df_benchmark, benchmark, method = method)
    alfa_hist = alpha(df, df_benchmark, benchmark, method = method)

    # Organizando os resultados em uma tabela
    ## 1) Mudando o nome das colunas dos dfs de alfa calculados
    for col_nova, col_antiga in zip(df.columns, alfa_hist.columns):
        alfa_12_meses = alfa_12_meses.rename(columns = {col_antiga: col_nova})
        alfa_24_meses = alfa_24_meses.rename(columns = {col_antiga: col_nova})
        alfa_hist = alfa_hist.rename(columns = {col_antiga: col_nova})

    ## 2) Criando uma tabela com os valores de alfa calculados
    df_table = pd.DataFrame()
    for col in alfa_hist:
        valor_alfa_12 = alfa_12_meses[col].tail(1)
        valor_alfa_24 = alfa_24_meses[col].tail(1)
        valor_hist = alfa_hist[col].tail(1)
        df_table.loc[col, '12m'] = valor_alfa_12.iloc[0]
        df_table.loc[col, '24m'] = valor_alfa_24.iloc[0]
        df_table.loc[col, 'Histórico'] = valor_hist.iloc[0]

    ## 3) Formatando os valores da tabela df_table criada
    #df_table = df_table.applymap(lambda x: "{:.2%}".format(x))
    df_table.loc['Média'] = df_table.mean()
    df_table.loc['Máx'] = df_table.max()
    df_table.loc['Min'] = df_table.min()

    return df_table

# Alpha Histórico
def alpha_historico(df_cotas, df_benchmark, bench, method = None):
    '''Retorna um dataframe com o alfa diário histórico;
    df_cotas = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks, 
    bench = string com o nome do benchmark desejado;
    method = define o método ('retorno diario': alpha com base no retorno diario;
    'retorno acumulado' ou vazio: alpha com base em retorno acumulado)'''
    df = df_cotas.copy()
    # 1) Add colunas do free risk rate
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    for col in df.columns:
        data = primeiras_datas_validas[col]
        df[f'{col} - {bench}'] = df_benchmark.loc[data:, bench]

    # 2) Criando um dataframe com os dados do free risk rate
    bench_columns = df.filter(like=f'{bench}').columns
    df_bench = pd.DataFrame()
    for i in bench_columns:
        df_bench[i] = df[i]
        df = df.drop(i, axis = 1)

    if method == None:
        # 3) Retorno acumulado anualizado (ok)
        df_ret_acum =  retorno_acumulado_anualizado(df)
        df_ret_bench = retorno_acumulado_anualizado(df_bench)

    if method != None:
        df_ret_acum =  retorno_diario(df, 1)
        df_ret_bench = retorno_diario(df_bench, 1)


    # 4) Cálculo do alpha
    alfa = pd.DataFrame()
    for col in df.columns:
        alfa[col] = ( 1 + df_ret_acum[col]) / (1 + df_ret_bench[f'{col} - {bench}']) - 1
    return alfa

#Beta Rolling
def beta_rolling(df_cotas, df_benchmark, lista_bench, focus, janela):
    beta = pd.DataFrame()
    aux = df_cotas.copy()

    for i in lista_bench:
        aux[i] = df_benchmark[i]
    df_retornos = retorno_diario(df_cotas)
    df_retornos_bench = retorno_diario(aux)

    for bench in lista_bench:
        covar = df_retornos[focus].rolling(janela).cov(df_retornos_bench[bench])
        vari = df_retornos_bench[bench].rolling(janela).var()
        beta[f'Beta {focus} x {bench}'] = covar.div(vari, axis = 0)

    return beta

# Máximo Drawdown)
def mdd(df_cotas):
    '''Retorna um dataframe de MDD;
    df_cotas: dataframe de cotas'''
    df_ret_acum = retorno_acumulado_cumprod(df_cotas, 0, 'with nan')
    df_ret_acum_n_normalizado = df_ret_acum + 1
    max = df_ret_acum_n_normalizado.cummax()
    drawdown = df_ret_acum_n_normalizado / max - 1
    return drawdown

def mdd_alpha(df_cotas, df_benchmark, bench):
    '''Retorna um dataframe de MDD do alpha;
    df_cotas = dataframe de cotas;
    df_benchmark = dataframe de cotas dos benchmarks;
    bench = string com o nome do benchmark de referência para o cálculo do alpha'''
    df_alpha = alpha(df_cotas, df_benchmark, bench, 'retorno acumulado')
    df_alpha_n_normalizado = df_alpha + 1
    max = df_alpha_n_normalizado.cummax()
    drawdown = df_alpha_n_normalizado / max - 1
    return drawdown

def recuperaçao_drawdown(df_drawdown, focus, dias_drawdown_relevante = None):
    '''A função retorna um dataframe com a quantidade de dias para a recuperação dos drawdowns;
    df_drawdown = df de max drawdown;
    focus = string com o nome do ativo ponto focal da análise de recuperação de drawdown;
    dias_drawdown_relevante = número inteiro que representa quantidade de dias mínimo para o drawdown ser considerado relevante'''
    # Inicializa uma lista para armazenar os resultados
    recuperacao = pd.DataFrame()
    
    # Inicializa variáveis para controlar o início do drawdown e a recuperação
    inicio_drawdown = None
    data_inicio_mdd = None
    fim_drawdown = None
    data_final_mdd = None
    
    for col in df_drawdown.columns:
        serie = df_drawdown[col]

        # Itera sobre a série
        for i in range(1, len(serie)):
            # Se estamos no início de um drawdown
            if serie[i] < 0 and serie[i-1] >= 0:
                inicio_drawdown = i
                data_inicio_mdd = df_drawdown.index[i].date()
            # Se estamos no fim de um drawdown (recuperação)
            elif serie[i] == 0 and serie[i-1] < 0:
                fim_drawdown = i
                data_final_mdd = df_drawdown.index[i].date()
                # Se um drawdown foi identificado anteriormente
                if inicio_drawdown is not None:
                    # Calcula a duração do drawdown
                    recuperacao.loc[f'{data_inicio_mdd} até {data_final_mdd}', col] = fim_drawdown - inicio_drawdown
                    # Reseta o início do drawdown para None
                    inicio_drawdown = None
    recuperacao_aux = pd.DataFrame()
    recuperacao_aux[focus] = recuperacao[focus]
    recuperacao_aux = recuperacao_aux[recuperacao_aux >= dias_drawdown_relevante]
    recuperacao_aux = recuperacao_aux.dropna()
    recuperacao_aux.loc['Média'] = recuperacao_aux.mean()
    recuperacao_aux.loc['Máx'] = recuperacao_aux.max()
    recuperacao_aux.loc['Mín'] = recuperacao_aux.min()
    return recuperacao_aux

#Correlação Rolling
def corr_rolling(df, janela, focus, referencia_corr):
    '''Retorna um dataframe de correlção rolling;
    df = dataframe de cotas;
    janela = número de janela rolling;
    focus = fundo foco da análise de correlação;
    referencia_corr = fundo ou bench que deseja-se calcular a correlação em relação ao focus'''
    df_retornos = retorno_diario(df, 1)
    roll_corr = pd.DataFrame()
    roll_corr[f'Corr {focus} x {referencia_corr}'] = df_retornos[focus].rolling(janela).corr(df_retornos[referencia_corr])
    return roll_corr

# Correlação Ano a Ano 
def corr_ano_a_ano(df, focus):
    '''
    A função retorna um dataframe com nível de correlação do fundo focus em relação aos demais fundos presentes no dataframe df fornecido;
    df = df de cotas;
    focus = string com o nome do fundo foco da análise;
    '''
    # 1) Tratando o df para que ele comece a partir da data do fundo mais recente
    primeiras_datas_validas = df.apply(pd.Series.first_valid_index)
    data_mais_recente = primeiras_datas_validas.max()
    df = df.loc[data_mais_recente:] # tr

    # 2) Calculando o retorno diário do df
    df_retornos = retorno_diario(df) 

    # 3) Organizando os dados por ano
    lista_df_anos = df_por_ano(df_retornos, 'lista') # Lista dos anos presentes no df_retornos
    dict_df_anos = df_por_ano(df_retornos, 'dict') # Retornos dos fundos separados por ano

    # 4) Calculando a Correlação de cada um dos dataframes de retorno separados por ano
    dict_corr_ano = {}
    for i in lista_df_anos:
        df_sample = dict_df_anos[i] # Pegando os dataframes de retorno dos fundos de cada ano
        corr_df = df_sample.corr() # Calculando a correlação de cada um dos dataframes de retorno dos fundos em cada ano
        #corr_df = check_nan_corr(corr_df)# # Não utilizado
        dict_corr_ano[i] = corr_df # Guardando as matrizes de correlação calculadas em um novo dicionário, cada matrzi separada por ano

    # 5) Organizando os dados obtidos de correlação em uma tabela ano a ano
    tabela = pd.DataFrame()
    for i in lista_df_anos:
        tabela.loc[:, i] = dict_corr_ano[i][focus]
    tabela = tabela.drop(focus)
    print(f'Tabela de Correlação ano a ano para o fundo {focus}')
    return tabela

def exp_do_cotista(df):
    '''A função retorna um dataframe de experiência do cotista;
    df = dataframe de cotas'''
    last_line = df.tail(1)
    exp_cotista = pd.DataFrame(columns = df.columns, index = df.index)
    for col in df.columns:
            # Pegando cada coluna separadamente
            df_sample = pd.DataFrame()
            df_sample[col] = df[col]
            df_sample = df_sample.dropna()
            df_sample = df_sample.reset_index()
            # pegando a cota final de cada uma das colunas
            cota_final = last_line.iloc[0][col]
            for index in df_sample.index:
                data = df_sample.loc[index, 'Date'] # Pegando as datas presentes na coluna 'Date' de cada um dos df_samples
                cotas_i = df_sample.loc[index, col] # Pegando cada uma das cotas de cada coluna, começando com a primeira cota
                # Calculando a experiência do cotista e salvando no dataframe exp_cotista (index do dataframe exp_cotista está com as datas enquanto que dataframe o df_sample está com o index resetado)
                exp_cotista.loc[data, col] = ((cota_final)/(cotas_i)) ** (252/(len(df_sample) - index)) - 1

    return exp_cotista