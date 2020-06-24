#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[32]:


import pandas as pd
import numpy as np


# In[55]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[34]:


def q1():
    return (black_friday.shape[0], black_friday.shape[1])
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[45]:


def q2():
    #return int(black_friday.groupby(['Gender','Age'])['User_ID'].nunique()['F','26-35'])
    return int(black_friday.groupby(['Gender'])['Age'].value_counts()['F','26-35']) 
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[36]:


def q3():
    return black_friday['User_ID'].nunique() 
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[37]:


def q4():
    return black_friday.dtypes.nunique() 
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[59]:


def q5():
    #return sum(black_friday.isna().sum())/(black_friday.shape[0]*black_friday.shape[1])
    return (black_friday.isna().sum()/black_friday.shape[0]).max() 
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[39]:


def q6():
    return int(black_friday.isna().sum().max()) 
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[40]:


def q7():
    return black_friday['Product_Category_3'].mode()[0]
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[41]:


def q8():
    Purchase = (black_friday['Purchase']-black_friday['Purchase'].min()) / (black_friday['Purchase'].max()-black_friday['Purchase'].min())
    return Purchase.mean()
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[42]:


def q9():
    Purchase_padrao = (black_friday['Purchase']-black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    maior =  Purchase_padrao > -1 
    menor = Purchase_padrao < 1
    return int(Purchase_padrao[maior & menor].count())
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[43]:


def q10():
    Product_Category_2_NA= black_friday['Product_Category_2']==np.nan
    Product_Category_3_NA= black_friday['Product_Category_3']==np.nan
    nullna = Product_Category_2_NA == Product_Category_3_NA
    return bool(nullna.unique()[0])
    pass

