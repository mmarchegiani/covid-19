import pandas as pd
import numpy as np

url_total  = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv'
url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
url_recovs = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv'

def covid_country(country = 'Total', url = url_total):
    df = pd.read_csv(url)
    df.rename(columns={"Country/Region" : "Country", "Province/State" : "Province"}, inplace=True)
    if country == 'Total':
        return df
    else:
        df_country = df[df['Country'].str.match(country)]
        return df_country
    
def get_cases(df_country, period = '1/22/20'):
    return df_country.loc[:, '1/22/20':].values[0]