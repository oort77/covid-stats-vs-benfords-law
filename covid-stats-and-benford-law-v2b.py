# COVID statistical data (new cases/new deaths) vs. Benford theorem
# -*- coding: utf-8 -*-
#  File: Covid stats and Benford law v2b.py
#  Project: 'Covid stats vs Benford's law'
#  Created by Gennady Matveev (gm@og.ly) on 02-07-2021.
#  Copyright 2021. All rights reserved.

# Import modules
import streamlit as st
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import requests
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import norm
from fuzzywuzzy import fuzz


plt.style.use("seaborn")  # ggplot

st.set_page_config(page_title="COVID statistics and Benford's Law",
                   page_icon='images/icon.ico',
                   initial_sidebar_state='auto')  # layout='wide',

# Auxiliary function for fuzzy search


def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

# Check if you already have today's data


def have_todays_data(file):
    if os.path.exists(file):
        (_, _, _, _, _, _, _, atime, mtime, ctime) = os.stat(file)
        modif_time_string = time.ctime(atime)
        modif = datetime.strptime(modif_time_string, '%a %b %d %H:%M:%S %Y')
        today = datetime.today()
        if modif.day == today.day and modif.month == today.month:
            return True
        else:
            return False
    else:
        return False


st.title("COVID statistics and Benford's Law")
st.markdown("""---""")

# Fetch and preprocess data
# cases url
url_cases = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
# deaths url
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

switch = st.sidebar.selectbox('Choose data', ('Cases', 'Deaths'), index=0)
if switch == 'Deaths':
    url = url_deaths
    csv_file = 'deaths_stat.csv'  # '/home/gm/notebooks/covid/deaths_stat.csv'
    description = 'deaths'
else:
    url = url_cases
    csv_file = 'cases_stat.csv'  # '/home/gm/notebooks/covid/cases_stat.csv'
    description = 'cases'

# Download only newer data
if have_todays_data(csv_file):
    raw_data = pd.read_csv(csv_file)
else:
    response = requests.get(url)
    with open(csv_file, 'wb') as f:
        f.write(response.content)
        raw_data = pd.read_csv(csv_file)

countries = raw_data['Country/Region'].unique()
raw_data.drop(columns=['Lat', 'Long', 'Province/State'], inplace=True)

# Fuzzy country name input
exists = False
while not exists:
    country = st.sidebar.text_input(
        'Country',
        value='Russia',
        help='Minor typos in countries names are forgiven)')
    if country.title() in countries:
        exists = True
    else:
        country = countries[argmax(
            list(
                map(lambda x: fuzz.token_set_ratio(x, country.title()),
                    countries)))]
        exists = True

raw_data = raw_data.groupby('Country/Region').sum()
raw_data = raw_data.apply(lambda x: x.diff(1), axis=1).fillna(0)

# Set p-value
p = float(st.sidebar.selectbox('Choose p-value', ('0.01', '0.05'), index=0))

# Calculate first digit occurences
raw_data = raw_data.applymap(lambda x: int(str(abs(x))[0]))
counts = raw_data.apply(lambda x: pd.value_counts(x), axis=1)
counts.drop(columns=0, inplace=True)
counts = counts.apply(lambda x: x / np.sum(x), axis=1)

ru_counts = counts.loc[country]  # .iloc[100]  #214
world_counts = counts.apply(np.mean, axis=0)

# Plot Benford's law vs averaged world statistics of occurences
fig1 = plt.figure(figsize=(12, 5))
plt.plot(
    world_counts,
    label=f'Daily new {description} in all countries (first digit)')
log = [np.log10(1 + 1 / i) for i in range(1, 10)]
plt.plot(range(1, 10), log, color='green', label="Benford's law: log(1+1/n)")
plt.legend(fontsize=14)
# st.write(f'''
#     Frequencies of the first digit occurence of daily new 
#     COVID {description} in all countries vs Benford's law:
#     ''')
st.write(f'''
    Frequencies of the first digit occurence of daily new 
    COVID {description} in all countries vs [Benford's
    law:](https://en.wikipedia.org/wiki/Benford's_law)
    ''')
plt.show()
st.write(fig1)

st.markdown(f'''
    Observation: distribution is well aligned with Benford's law.
    ''')
st.markdown("""---""")

# Plot all
fig2 = plt.figure(figsize=(12, 7))
sns.boxplot(data=counts)
sns.lineplot(x=range(9),
             y=ru_counts,
             color='red',
             marker='o',
             label=f'Daily new {description} in {country} (first digit)')
log = [np.log10(1 + 1 / i) for i in range(1, 10)]
plt.plot(log,
         color='green',
         linestyle='-',
         linewidth=2,
         label="Benford's law: log(1+1/n)")
plt.legend(fontsize=14)
st.markdown(f'''
    Frequencies of the first digit occurence of daily new 
    COVID {description} in **{country}** vs all other countries:
    ''')
plt.show()
st.write(fig2)

st.markdown("""---""")
st.markdown('**Null hypothesis:**')
st.write(
    'First digit of new daily COVID cases in the country is distributed similarly to first digits of new daily COVID cases in all other countries.'
)

# Main function

def check_null(country_counts, df_stat, p):
    check_results = np.zeros(9)
    for n in df_stat.columns:
        prob = norm.cdf(country_counts[n],
                        loc=df_stat.iloc[1, n - 1],
                        scale=df_stat.iloc[2, n - 1])  # /np.sqrt(400*0.15))
        check_results[n - 1] += (prob < p or prob > 1 - p)
    strikes = sum(check_results)
    return check_results, strikes


# Check null hypothesis for the chosen country
check_result, no_strikes = check_null(ru_counts, counts.describe(), p)
check_result = [int(i) for i in check_result]
check_df = pd.DataFrame(check_result, index=range(1, 10), columns=['Rejected'])

st.markdown("""---""")
st.markdown(f'Check null hypothesis for **{country}**:')
st.markdown(
    f'Null hypothesis is rejected for {int(no_strikes)} digit(s).')
st.write(check_df.T)

# Plot null hypothesis rejection histogram

counts_desc = counts.describe()
counts['strikes'] = counts.apply(lambda x: check_null(x, counts_desc, p)[1],
                                 axis=1)

# Format dataframe for output
x = counts[counts['strikes'] >= 2].sort_values(by='strikes', ascending=False)
x.columns = x.columns.astype("str")
cols = ['strikes'] + [col for col in x if col != 'strikes']
x = x[cols]
x.rename(columns={'strikes': 'Rejections #'}, inplace=True)

st.markdown("""---""")
st.write(
    f'Distribution of countries/regions by the number of null hypothesis rejections, p = {p}:'
)
fig3 = plt.figure(figsize=(12, 5))
fig3 = px.histogram(counts, x="strikes", width=1200, height=500)
#fig3.add_annotation(x=4, y=100,
#            text=f'p = {p}',
#            showarrow=False,
#            yshift=10)
fig3.update_layout(
    #    title_text='Sampled Results', # title of plot
    xaxis_title_text='Rejections #',
    yaxis_title_text='Count',
    bargap=0.5
)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("""---""")

st.write("COVID statistics 'champions':")
st.write(
    x.style.format("{:.2}", na_rep="-").format({'Rejections #': "{:.0f}"}))
