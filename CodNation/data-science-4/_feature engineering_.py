#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[49]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[50]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[51]:


countries = pd.read_csv("countries.csv",decimal=',')


# In[52]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[214]:


# Sua análise começa aqui.
#countries.dtypes
#countries.isna().sum()
#countries.shape


# In[215]:


#Retirando os espaços das variáveis Country e Region
countries.Country = pd.Series(countries.Country).str.strip()
countries.Region  = pd.Series(countries.Region).str.strip()


# In[154]:


#countries.Region.value_counts().plot.bar()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[216]:


lista = list(pd.DataFrame(countries.Region.unique()).sort_values(by=[0]).iloc[:,0])


# In[217]:


def q1():
    return lista
    pass


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[218]:


discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

discretizer.fit(countries[["Pop_density"]])

Pop_density_bins = discretizer.transform(countries[["Pop_density"]])

#Pop_density_bins[:5]
#227-pd.DataFrame(Pop_density_bins)[0].value_counts().sum()*0.90


# In[45]:


def q2():
    return int(sum(Pop_density_bins[:, 0]==9)) 
    pass


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[222]:


one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)

region_encoded = one_hot_encoder.fit_transform(countries[["Region"]])

climate_encoded = one_hot_encoder.fit_transform(countries[["Climate"]].fillna(0))

print(region_encoded.shape[1]+climate_encoded.shape[1])


# In[223]:


def q3():
    return int(region_encoded.shape[1]+climate_encoded.shape[1])
    pass


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[224]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[225]:


var_num = var_num = countries.select_dtypes(exclude='object')

pipe = Pipeline(steps=[ ("Preencher", SimpleImputer(strategy="median")), ("standard_scaler", StandardScaler())])

pipe.fit(var_num);


# In[226]:


pipe_transform = pipe.transform([test_country[2:]])
num_arable = pipe_transform[:,var_num.columns.get_loc('Arable')]
num_arable.round(3)


# In[227]:


def q4():
    return float(num_arable.round(3))
    pass


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[53]:


countries[['Net_migration']].fillna(countries[['Net_migration']].mean())
Q1 =countries[['Net_migration']].describe().loc['25%','Net_migration']
Q3 = countries[['Net_migration']].describe().loc['75%','Net_migration']
IQR = Q3-Q1
outliers_abaixo=0 
outliers_acima=0
for i in range(0,len(countries[['Net_migration']])):
    if (countries['Net_migration'][i] < (Q1 - 1.5*IQR)):
        outliers_abaixo = outliers_abaixo + 1
    if (countries['Net_migration'][i] > (Q3 + 1.5*IQR)):
        outliers_acima = outliers_acima + 1
print('outliers_abaixo:',outliers_abaixo,'outliers_acima:',outliers_acima)            


# In[55]:


#countries[['Net_migration']].boxplot()
Net_migration = (int(outliers_abaixo), int(outliers_acima), bool(False))


# In[24]:


#countries[['Net_migration']].fillna(countries[['Net_migration']].mean(),inplace=True)
#q1 = Net_migration.quantile(0.25)
#q3 = Net_migration.quantile(0.75)
#iqr = q3 - q1
#outliers_abaixo=0 
#outliers_acima=0
#for i in range(0,len(countries[['Net_migration']])):
#    if (countries['Net_migration'][i] < (Q1 - 1.5*IQR)):
#        outliers_abaixo = outliers_abaixo + 1
#    if (countries['Net_migration'][i] > (Q3 + 1.5*IQR)):
#        outliers_acima = outliers_acima + 1
#print('outliers_abaixo:',outliers_abaixo,'outliers_acima:',outliers_acima)   


# In[56]:


def q5():
    return tuple(Net_migration)
    pass


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[19]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']

newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[20]:


len(newsgroups.data)


# In[21]:


count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(newsgroups.data)


# In[47]:


num_phone = counts[:,count_vectorizer.vocabulary_['phone']].sum()


# In[ ]:


def q6():
    return int(num_phone)
    pass


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[38]:


tfidf_vectorizer = TfidfVectorizer()

tfidf_vectorizer.fit(newsgroups.data)

newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups.data)


# In[43]:


TF_IDF = newsgroups_tfidf_vectorized[:,tfidf_vectorizer.vocabulary_['phone']].sum().round(3)


# In[44]:


def q7():
    return float(TF_IDF)
    pass

