#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[161]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[162]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[163]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[164]:


# Sua análise da parte 1 começa aqui.


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[165]:


def q1():
    return ((dataframe.describe().loc['25%','normal']-dataframe.describe().loc['25%','binomial']).round(3), (dataframe.describe().loc['50%','normal']-dataframe.describe().loc['50%','binomial']).round(3), (dataframe.describe().loc['75%','normal']-dataframe.describe().loc['75%','binomial']).round(3))
    pass


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[180]:


def q2():
    x_s_inf = dataframe['normal'].mean() - dataframe['normal'].std()
    x_s_sup = dataframe['normal'].mean() + dataframe['normal'].std()
    ecdf    = ECDF(dataframe['normal'])
    array   = ecdf([x_s_inf,x_s_sup]) 
    return float((array[1]-array[0]).round(3))
    pass


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[167]:


def q3():
    return ((dataframe.describe().loc['mean','binomial']-dataframe.describe().loc['mean','normal']).round(3),round(dataframe['binomial'].var()-dataframe['normal'].var(),3))
    pass


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[196]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[169]:


# Sua análise da parte 2 começa aqui.


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[251]:


def q4():
    df = stars[stars['target']==0]['mean_profile']
    #Padronizando
    false_pulsar_mean_profile_standardized = (df-df.mean())/df.std()
    #Usando norm.ppf
    quantis_80 = sct.norm.ppf(0.80, loc=false_pulsar_mean_profile_standardized.mean(), scale=false_pulsar_mean_profile_standardized.std())
    quantis_90 = sct.norm.ppf(0.90, loc=false_pulsar_mean_profile_standardized.mean(), scale=false_pulsar_mean_profile_standardized.std())
    quantis_95 = sct.norm.ppf(0.95, loc=false_pulsar_mean_profile_standardized.mean(), scale=false_pulsar_mean_profile_standardized.std())
    #Usando ECDF 
    ecdf   = ECDF(false_pulsar_mean_profile_standardized)
    array  = ecdf([quantis_80,quantis_90,quantis_95])
    cdf_80 = array[0] 
    cdf_90 = array[1] 
    cdf_95 = array[2]
    return (cdf_80.round(3),cdf_90.round(3),cdf_95.round(3))
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[253]:


def q5():
    df = stars[stars['target']==0]['mean_profile']
    #Padronizando
    false_pulsar_mean_profile_standardized = (df-df.mean())/df.std()
    #Quantis
    Q1=false_pulsar_mean_profile_standardized.describe().loc['25%'] 
    Q2=false_pulsar_mean_profile_standardized.describe().loc['50%']
    Q3=false_pulsar_mean_profile_standardized.describe().loc['75%']
    Q1_normal=sct.norm.ppf(0.25,loc=0, scale=1)
    Q2_normal=sct.norm.ppf(0.50,loc=0, scale=1)
    Q3_normal=sct.norm.ppf(0.75,loc=0, scale=1)
    return((Q1-Q1_normal).round(3),(Q2-Q2_normal).round(3),(Q3-Q3_normal).round(3))
    pass


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
