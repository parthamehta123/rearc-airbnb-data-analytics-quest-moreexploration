# ------------------------------------------------- ------------------------------------------------- WE IMPORT
# LIBRARIES------------------------------------------------- --------------------------------------------

# plot graphics and visualization
import matplotlib.pyplot as plt
import numpy as np
# basic libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
# streamlit
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# machine learning libraries for linear regression
from sklearn.model_selection import train_test_split
# text editor
from wordcloud import WordCloud

# interactive maps
# import chart_studio.plotly as py

# ------------------------------------------------ --------------------------------------------------PREPROCESSAMIENTO -------------------------------------------------- -------------------------------------------------- ------
# read the dataframe that we are going to use
listings = pd.read_csv('detailed_listings.csv')
listingsnulls = pd.read_csv('detailed_listings.csv')

# add the entire preprocessing

# list of columns to remove that exceed 30% of null values
columns_to_drop = ["neighbourhood_group_cleansed", "calendar_updated", "bathrooms", "license", "host_about",
                   "host_neighbourhood", "neighborhood_overview", "neighbourhood"]
listings.drop(columns_to_drop, axis=1, inplace=True)


# create a function to fill in the null values ​​by the mean or the mode depending on the type of data
def fillna(df, columns):
    """fillna fills null values ​​in a DataFrame with the mean or mode depending on the column type.
    :param df: DataFrame containing the data.
    :param columns: List with the names of the columns that contain
    :return: Returns dataframe with null values padded.
    """
    for column in columns:
        if (df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            number = df[column].mean()
        else:
            number = df[column].mode()[0]
        df[column] = df[column].fillna(number)
    return df


listings = fillna(listings, listings.columns)


# create a function to remove the outliers
def outliers_repair(df):
    """outliers_repair is used to identify and repair outliers in a given DataFrame.
    :param listings: DataFrame for which outliers should be fixed.
    :returns: dataFrame without outliers.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

    dataset_wo_outliers = df.copy()
    dataset_wo_outliers = pd.DataFrame(df)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode = df[col].mode()[0]
            dataset_wo_outliers.loc[outliers[col], col] = mode
        else:
            mean = df[col].mean()
            dataset_wo_outliers.loc[outliers[col], col] = mean
    return dataset_wo_outliers


listings = pd.DataFrame(outliers_repair(listings))


# make a function to remove the $ symbol from the necessary columns
def remove_sign(df, column):
    """remove_sign is used to remove a symbol from in this case a number
    :param listings: dataframe we are using
    :param price: column we want to use
    :return: dataframe with the text removed from the column
    """
    listings['price'] = listings['price'].apply(lambda x: x.replace("$", ""))
    return listings


listings = remove_sign(listings, 'price')


# make a function to remove the commas and convert the price to float
def fix_price_column(df):
    """fix_price_column removes the commas from the column we select from our dataframe and changes the values ​​to float
    :param df: Dataframe on which we are going to use this function
    :return: the corrected value within the assigned column
    """
    listings['price'] = listings['price'].str.replace(',', '')
    listings['price'] = listings['price'].astype(float)
    return listings


listings = fix_price_column(listings)


# make a function to eliminate a length that gives problems when rendering the maps
def drop_rows(df, column, value):
    """drop rows removes the value we want from a row from a column
    :param df: dataframe we are going to use
    :param column: column that will have the values ​​of the rows that we want to delete
    :param value: value we want to blur
    :return: returns the dataframe
    """
    df = df[df[column] != value]
    return df


listings = drop_rows(listings, "longitude", -122.43018616682234)

# create a lambda function that allows us to group the distribution at normal prices
listings['price'] = listings['price'].apply(
    lambda x: int(x / 1000) if x >= 10000 else (int(x / 100) if x >= 1000 else x))


# create a function to remove the outliers from the price column
def remove_outliers(df, column):
    """this function removes outlier values ​​within a column from our def
        :param df: llama al dataframe
        :param column: column to remove outliers
        :return: returns the df without outliers in the selected column
    """
    df_clean = df[df[column].notnull()]
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[(df[column] < mean - 2 * std) | (df[column] > mean + 2 * std)].index  # std*2 is twice the standard deviation used to define a range of "normal" values ​​for a variable
    df.drop(outliers, inplace=True)
    return df


listings = remove_outliers(listings, 'price')

# we group the neighborhoods in their districts for a better reading and we create a new column inside the dataframe
listings['districts'] = ''
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Pacific Heights', 'Nob Hill', 'Presidio Heights', 'Russian Hill']), 'districts'] = 'Central/downtown'
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Western Addition', 'Haight Ashbury', 'Glen Park', 'Downtown/Civic Center', 'Financial District', 'Marina',
     'North Beach', 'Chinatown']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Bernal Heights', 'Bayview', 'Excelsior', 'Outer Mission', 'Inner Sunset', 'Visitacion Valley', 'Crocker Amazon',
     'Ocean View', 'Parkside']), 'districts'] = 'Bernal Heights/Bayview and beyond (southeast)'
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Mission', 'Castro/Upper Market', 'Potrero Hill', 'South of Market',
     'Outer Sunset']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Outer Richmond', 'Inner Richmond', 'West of Twin Peaks', 'Noe Valley', 'Twin Peaks', 'Golden Gate Park',
     'Presidio', 'Lakeshore', 'Diamond Heights', 'Seacliff']), 'districts'] = 'Richmond'
listings.loc[listings['neighbourhood_cleansed'].isin(
    ['Inner Richmond', 'Inner Sunset', 'Parkside', 'Outer Sunset']), 'districts'] = 'Sunset'

# ------------------------------------------------- -----------------------------------WE START THE APP ------------ -------------------------------------------------- ---------------------------------
st.set_page_config(page_title='SAN FRANCISCO', layout='centered', page_icon="✈️")
image = Image.open(
    'puentewide.png')
st.image(image, width=800)
st.write('San Francisco Bridge: Image created with DALL-E-2')
st.title('EDA AIRBNB: SAN FRANCISCO')

# create the tabs that will divide our app
tabs = st.tabs(
    ['INSIDE AIRBNB', 'PREPROCESSING', 'EDA', 'CONCLUSIONS', 'LINEAR REGRESSION'])

# ------------------------------------------------- ------------------------------FIRST TAB: INSIDE AIR BNB-------------- -------------------------------------------------- --------------------
tab_plots = tabs[0]

# show the Inside Airbnb page
with tab_plots:
    st.header('Inside Airbnb (Web)')
    st.write(
        f'<iframe src="http://insideairbnb.com/san-francisco/" width="800" height="600" style="overflow:auto"></iframe>',
        unsafe_allow_html=True)
    url2 = "http://insideairbnb.com/san-francisco/"
    st.markdown("[Inside Airbnb](%s)" % url2)

# show the dataframe extracted from the Inside Airbnb page
with tab_plots:
    st.header('DataSet')
    st.markdown('DataSet before preprocessing, downloaded from the Inside Airbnb website')
    st.write(listingsnulls)
    url3 = "http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"
    st.markdown("[Download the CSV](%s)" % url3)

# ------------------------------------------------- ------------------------------SECOND TAB: PREPROCESSING---------------- -------------------------------------------------- -------------------
tab_plots = tabs[1]

# import libraries
with tab_plots:
    st.header('Preprocessing completed')
    st.subheader('*Preprocessing has been one of the main parts of this work to be able to perform the EDA*')
    st.markdown('Libraries needed to carry out this project (Requirements)')
    code = """
    # basic libraries
    import us
    import re
    import requests
    import pandas as pd
    import numpy as np
    # machine learning libraries for linear regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
    from sklearn import preprocessing
    # interactive maps
    import leaf
    from folium.plugins import FastMarkerCluster
    # plot graphics and visualization
    import matplotlib.pyplot as plt
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.offline import plot
    import chart_studio.plotly as py
    # streamlit
    import streamlit as st
    import streamlit.components.v1 as components
    import google 
    from PIL import Image
    from pyngrok import ngrok
    from IPython.display import display
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    import urllib.request
    import unicodedata
    from unicodedata import name
    import plotly.io as pio
    # text editor
    from wordcloud import WordCloud
    """
    st.code(code, language='python')

# we create a function to compare the datasets that are the same when downloading these
with tab_plots:
    st.header('Preprocessing complete')
    st.subheader('*Preprocessing has been one of the main parts of this work to be able to perform the EDA*')
    st.markdown(
        'When downloading the datasets from the insideairbnb page, we verify that we have datasets with the same name. With which we are going to compare if they also have equal columns.')
    code = """
    def columns_compare(df1, df2):
        #columns_compare takes two dataframes as inputs and compares the columns of both dataframes.
        #:param: df1 (DataFrame): Primer DataFrame
        #param: df2(DataFrame): Second DataFrame
        #return: List with the same columns between the two data frames.
    columns_df1 = df1.columns.tolist()
    columns_df2 = df2.columns.tolist()
    equal_columns = list(set(columns_df1) & set(columns_df2))
    return equal_columns
equal_columns = columns_compare(listings, listings1)
print("Equal Columns: ",equal_columns)
"""
    st.code(code, language='python')

# compare the columns of the datasets with the same name, if they have the same values ​​in order to perform a merge
with tab_plots:
    st.markdown(
        'Having columns with the same names, we compare a column to see if all the values ​​are similar, so we can choose whether to merge any and join the dataframes.')
    code = """
    def value_colums(df1, df2):
        #value_colums takes two dataframes as inputs and checks if the column of both dataframes have the same values.
        #:param: df1 (DataFrame): Primer DataFrame
        #:param: df2 (DataFrame): Second DataFrame
        #:return: True if the values ​​are equal, False if they are not.
    values_df1 = df1['id'].values
    values_df2 = df2['id'].values
    if np.array_equal(values_df1, values_df2):
        return True
    else:
        diferents = np.where(values_df1 != values_df2)[0]
        print("Different values:", values_df2[different])
        return False
value_colums(listings, listings1)    
"""
    st.code(code, language='python')

# create a variable to know the % of null values
with tab_plots:
    st.markdown('We create a variable percent_missing and with it we order the null values ​​by percentage.')
    code = """
    percent_missing = listings.isnull().sum() * 100 / len(listings)
    percent_missing.sort_values(ascending = False).head()
"""
    st.code(code, language='python')

# make a graph on the nulls inside the dataset
with tab_plots:
    nulls = listingsnulls.isnull().sum()
    fig = go.Figure(go.Bar(x=nulls.index, y=nulls))
    fig.update_layout(template="plotly_dark", xaxis_title="Columns per index", yaxis_title="Total Nulls")
    fig.update_layout(xaxis=dict(title_standoff=10, tickangle=45))
    st.subheader('*Distribution of null values ​​by columns*')
    st.plotly_chart(fig)

# list of columns to remove that exceed 30% of null values
with tab_plots:
    st.markdown('We remove the columns that have more than 30%, we add an inplace = True, to fix this action.')
    code = """
    columns_to_drop = ["neighbourhood_group_cleansed", "calendar_updated", "bathrooms", "license", "host_about", "host_neighbourhood", "neighborhood_overview", "neighbourhood"]
    listings.drop(columns_to_drop, axis=1, inplace=True)
"""
    st.code(code, language='python')

# create a function to fill in the null values ​​by the mean or the mode depending on the type of data
with tab_plots:
    st.markdown('We convert the remaining null values, using the mean and mode as appropriate.')
    code = """
    def fillna(df, columns):
    #fillna fills null values ​​in a DataFrame with the mean or mode depending on the column type.
    #:param df: DataFrame containing the data.
    #:param columns: List with the names of the columns that contain
    #:return: Returns dataframe with null values ​​padded.
    for column in columns:
        if(df[column].dtype == 'float64' or df[column].dtype == 'int64'):
            number = df[column].mean()
        else:
            number = df[column].mode()[0]
        df[column]= df[column].fillna(number)
    return df
listings = fillna(listings, listings.columns)
"""
    st.code(code, language='python')

# create a function to remove the outliers
with tab_plots:
    st.markdown('We use this function to remove outliers from our columns.')
    code = """
def outliers_repair (df):
    #outliers_repair is used to identify and repair outliers in a given DataFrame.
    #:param listings: DataFrame for which outliers should be fixed.
    #:returns: dataFrame without outliers.
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    dataset_wo_outliers = df.copy()
    dataset_wo_outliers = pd.DataFrame(df)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode = df[col].mode()[0]
            dataset_wo_outliers.loc[outliers[col], col] = mode
        else:
            mean = df[col].mean()
            dataset_wo_outliers.loc[outliers[col], col] = mean
    return dataset_wo_outliers
    listings = pd.DataFrame(outliers_repair(listings))
"""
    st.code(code, language='python')

# make a function to remove the $ symbol from the 'price' column
with tab_plots:
    st.markdown('In the price column we remove the $ symbol to be able to work better with this data.')
    code = """
    def remove_sign(df, column):
    #remove_sign is used to remove a symbol from in this case a number
    #:param listings: dataframe we are using
    #:param price: column we want to use
    #:return: dataframe with the text removed from the column
    listings['price'] = listings['price'].apply(lambda x: x.replace("$", ""))
    return listings
    listings = remove_sign(listings, 'price')
"""
    st.code(code, language='python')

# make a function to remove the commas and convert the price to float
with tab_plots:
    st.markdown(
        'We make a function to eliminate the commas and convert the price to float, since in the future it will give us problems to make the plots.')
    code = """
    def fix_price_column(df):
    #fix_price_column removes the commas from the column we select from our dataframe and changes the values ​​to float
    #:param df: Dataframe on which we are going to use this function
    #:return: the corrected value within the assigned column
    listings['price'] = listings['price'].str.replace(',','')
    listings['price'] = listings['price'].astype(float)
    return listings
listings = fix_price_column(listings)
"""
    st.code(code, language='python')

# make a function to eliminate a length that gives problems when rendering the maps
with tab_plots:
    st.markdown(
        'When making the maps, we realize that a coordinate within the longitude gives us problems, so we proceed to delete it.')
    code = """
    def drop_rows(df, column, value):
    #drop rows removes the value we want from a row from a column
    #:param df: dataframe we are going to use
    #:param column: column that will have the values ​​of the rows that we want to delete
    #:param value: value we want to blur
    #:return: returns the data frame
    df = df[df[column] != value]
    return df
    listings = drop_rows(listings, "longitude",  -122.43018616682234)
    """
    st.code(code, language='python')

    # create a lambda function that allows us to group the distribution at normal prices compared to the listings dataframe that comes prepared
    with tab_plots:
        st.markdown(
            'Create a lambda function that allows us to group the distribution at normal prices compared to the ready-made listings dataframe.')
    code = """listings['price'] = listings['price'].apply(lambda x: int(x/1000) if x >= 10000 else (int(x/100) if x >= 1000 else x))"""
    st.code(code, language='python')

# create a function to remove the outliers from the price column
with tab_plots:
    st.markdown(
        'We notice that we also have to remove several un-removed outliers in the price column, to get the charts right.')
    code = """
    def remove_outliers(df, column):
    #this function removes outlier values ​​within a column from our def
    #:param df: llama al dataframe
    #:param column: column to remove outliers
    #:return: returns the df without outliers in the selected column
    df_clean = df[df[column].notnull()]
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[(df[column] < mean - 2*std) | (df[column] > mean + 2*std)].index # std*2 is twice the standard deviation used to define a range of "normal" values ​​for a variable
    df.drop(outliers, inplace=True)
    return df
    listings = remove_outliers(listings, 'price')
"""
    st.code(code, language='python')

# we group the neighborhoods in their districts for a better reading and we create a new column inside the dataframe
with tab_plots:
    st.markdown('For a better reading of the data we group the neighborhoods into districts.')
    code = """
listings['districts'] = ''
listings.loc[listings['neighbourhood_cleansed'].isin(['Pacific Heights', 'Nob Hill', 'Presidio Heights', 'Russian Hill']), 'districts'] = 'Central/downtown'
listings.loc[listings['neighbourhood_cleansed'].isin(['Western Addition', 'Haight Ashbury', 'Glen Park', 'Downtown/Civic Center', 'Financial District', 'Marina', 'North Beach', 'Chinatown']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Bernal Heights', 'Bayview', 'Excelsior', 'Outer Mission', 'Inner Sunset', 'Visitacion Valley', 'Crocker Amazon', 'Ocean View', 'Parkside']), 'districts'] = 'Bernal Heights/Bayview and beyond (southeast)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Mission', 'Castro/Upper Market', 'Potrero Hill', 'South of Market', 'Outer Sunset']), 'districts'] = 'Upper Market and beyond (south central)'
listings.loc[listings['neighbourhood_cleansed'].isin(['Outer Richmond', 'Inner Richmond', 'West of Twin Peaks', 'Noe Valley', 'Twin Peaks', 'Golden Gate Park', 'Presidio', 'Lakeshore', 'Diamond Heights', 'Seacliff']), 'districts'] = 'Richmond'
listings.loc[listings['neighbourhood_cleansed'].isin(['Inner Richmond', 'Inner Sunset', 'Parkside', 'Outer Sunset']), 'districts'] = 'Sunset'
"""
    st.code(code, language='python')

# ------------------------------------------------- ------------------------------------ THIRD TAB: EDA---------- -------------------------------------------------- -----------
tab_plots = tabs[2]

# create a correlation matrix between several variables and a correlation map between those variables
with tab_plots:
    st.header('San Francisco Exploratory Analysis')
    st.subheader('*Correlation Table*')
    st.markdown(
        'We create a correlation matrix of these columns ["price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]')
    corr = listings[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                     'calculated_host_listings_count', 'availability_365']].corr()
    st.table(corr)
    st.subheader('*Correlation Matrix*')
    fig = px.imshow(corr, color_continuous_scale=px.colors.sequential.Jet, template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)
    st.markdown('- Number of minimum nights vs. Number of reviews: -0.3594.')
    st.markdown('- Number of revisions vs. Number of revisions per month: 0.3926.')
    st.markdown('- Price vs. Minimum number of nights: -0.1795.')
    st.markdown('- Price vs. Number of reviews: 0.0055.')
    st.markdown('- Price vs. Revisions per month: 0.0339.')

# We make a plot of the price by means of accommodations making a groupby according to district and guests
with tab_plots:
    st.subheader('*Distribution of rooms in San Francisco*')
    st.markdown('Partha Mehta Code')
    html = open("map.html", "r", encoding='utf-8').read()
    st.components.v1.html(html, height=600)

# plot foot of distribution in % of neighborhoods in San Francisco
with tab_plots:
    st.subheader('*Distribution in % of neighborhoods in San Francisco*')
    value_counts = listings['neighbourhood_cleansed'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hovertext=value_counts.index)])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# viewing the distribution on the pie chart, pass it to a map
with tab_plots:
    st.subheader('*Distribution of neighborhoods in San Francisco*')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='neighbourhood_cleansed',
                            size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r": 80, "t": 80, "l": 80, "b": 80})
    map.update_layout(width=800)
    map.update_layout(template="plotly_dark")
    st.plotly_chart(map)

# plot foot of the distribution in % of the districts in San Francisco
with tab_plots:
    st.subheader('*Distribution in % of the districts in San Francisco*')
    value_counts = listings['districts'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values, hovertext=value_counts.index)])
    fig.update_layout(template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# viewing the distribution on the pie chart, pass it to a map
with tab_plots:
    st.subheader('*Distribution of districts in San Francisco*')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='districts',
                            size_max=15, zoom=10, height=600)
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r": 80, "t": 80, "l": 80, "b": 80})
    map.update_layout(width=800)
    map.update_layout(template="plotly_dark")
    st.plotly_chart(map)

# We make a plot of the price by means of accommodations making a groupby according to district and guests
with tab_plots:
    st.subheader('*Average accommodation price: District and guests*')
    st.markdown('We found that in most districts guests choose 2- and 4-person rooms.')
    listings_accommodates = listings.groupby(['districts', 'accommodates'])['price'].mean().sort_values(
        ascending=True).reset_index()
    fig = px.bar(listings_accommodates, x='price', y='districts', color='accommodates', orientation='h',
                 labels={'price': 'Average price or Median price', 'neighbourhood_cleansed': 'Neighborhood or District',
                         'accommodates': 'Number of guests'}, template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# We create a graph of the average price of accommodation for 2 people in each neighborhood
with tab_plots:
    st.subheader('*Average price of accommodation for 2 people in each neighborhood*')
    st.markdown('The majority choice in San Francisco is for rooms for two people.')
    st.markdown('So we analyze that value for neighborhoods and districts.')
    st.markdown(
        'The neighborhood with the most accommodations for two people is ChinaTown followed by the Financial District')
    average = listings[listings['accommodates'] == 2]
    average = average.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=True)
    fig = go.Figure(data=[go.Bar(y=average.index, x=average.values, orientation='h')])
    fig.update_layout(xaxis_title='Average price or Median price', yaxis_title='Neighborhood or District', template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# create a graph of the average price of accommodation for 2 people in each district
with tab_plots:
    st.subheader('*Average price of accommodation for 2 people in each district*')
    st.markdown(
        'The district with the most accommodation is Upper Market, this district corresponds to the neighborhoods of ChinaTown and Financial District')
    average = listings[listings['accommodates'] == 2]
    average = average.groupby('districts')['price'].mean().sort_values(ascending=True)
    fig = go.Figure(data=[go.Bar(y=average.index, x=average.values, orientation='h')])
    fig.update_layout(xaxis_title='Average price or Median price', yaxis_title='District', template="plotly_dark")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# do a value counts to see how many room types there are and create a bar chart and plot it
with tab_plots:
    st.subheader('*Room types in San Francisco*')
    st.markdown('The most common room type is entire rooms or apartments')
    room_type_counts = listings.room_type.value_counts()
    fig = px.bar(room_type_counts, x=room_type_counts.index, y=room_type_counts.values,
                 template='plotly_dark',
                 color=room_type_counts.index,
                 color_continuous_scale='Plasma',
                 labels={'x': '', 'y': ''})
    fig.update_layout(xaxis_title="Type of room", yaxis_title="Total")
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# create a map that relates the distribution of prices and room types
with tab_plots:
    st.subheader('*Distribution of prices and room types*')
    st.markdown(
        'We note that the highest prices are found around "Golden Gate Park Avenue", "Post Street", "Bay Street, and "Market Street".')
    map = px.scatter_mapbox(listings, lat="latitude", lon="longitude",
                            opacity=1.0,
                            color='price',
                            color_continuous_scale=px.colors.sequential.Jet,
                            height=600, zoom=9.7,
                            text='room_type',
                            hover_name='name')
    map.update_layout(mapbox_style="open-street-map")
    map.update_layout(margin={"r": 80, "t": 80, "l": 80, "b": 80})
    map.update_layout(template="plotly_dark")
    map.update_layout(width=800)
    st.plotly_chart(map)

# We create price intervals to group the prices we find and make it easier to read and we distribute prices by neighborhood
with tab_plots:
    st.subheader('*Price distribution by neighborhood*')
    st.markdown('The neighborhood with the most expensive Airbnb is Western Addition')
    st.markdown('The neighborhood with the cheapest Airbnb is Bayview')
    listings['price_bin'] = pd.cut(listings["price"], [0, 100, 200, 300, 400, float('inf')],
                                   labels=["0-100", "100-200", "200-300", "300-400", "400-500"])
    # create a pivot table between the neighborhood column and the price
    pivot_table = listings.pivot_table(values='id', index='neighbourhood_cleansed', columns='price_bin',
                                       aggfunc='count')
    # make the bar chart first by introducing the pivot table
    data = []
    for i in pivot_table.index:
        data.append(go.Bar(x=pivot_table.columns, y=pivot_table.loc[i], name=i))
    # display the bar graph
    fig = go.Figure(data=data)
    fig.update_layout(template='plotly_dark', xaxis_title='Price', yaxis_title='Total number of houses', barmode='stack')
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# We create price intervals to group the prices we find and make it easier to read and we distribute prices by district
with tab_plots:
    st.subheader('*Price distribution by district*')
    st.markdown('The district with the most expensive Airbnbs is Upper Market')
    st.markdown('The district with the cheapest Airbnbs is Bernal Height')
    listings['price_bin'] = pd.cut(listings["price"], [0, 100, 200, 300, 400, float('inf')],
                                   labels=["0-100", "100-200", "200-300", "300-400", "400-500"])
    # create a pivot table between the neighborhood column and the price
    pivot_table2 = listings.pivot_table(values='id', index='districts', columns='price_bin', aggfunc='count')
    # make the bar chart first by introducing the pivot table
    data = []
    for i in pivot_table2.index:
        data.append(go.Bar(x=pivot_table2.columns, y=pivot_table2.loc[i], name=i))
    # display the bar graph
    fig = go.Figure(data=data)
    fig.update_layout(template='plotly_dark', xaxis_title='Price', yaxis_title='Total number of houses', barmode='stack')
    fig.update_layout(width=800)
    st.plotly_chart(fig)

# plot the number of days a particular host is available in a year
with tab_plots:
    st.subheader('*Number of days a particular host is available within a year*')
    st.markdown('Most are available 365 days a year')
    map = px.scatter_mapbox(listings, lat='latitude', lon='longitude', color='availability_365',
                            size_max=15, zoom=10, height=600, color_continuous_scale='viridis')
    map.update_layout(mapbox_style='open-street-map')
    map.update_layout(margin={"r": 80, "t": 80, "l": 80, "b": 80})
    map.update_layout(template="plotly_dark")
    map.update_layout(width=800)
    st.plotly_chart(map)

# We create a variable to see what possible illegal rentals we find
with tab_plots:
    st.subheader('*Laws that affect airbnb*')
    st.markdown("""In San Francisco, California, Airbnb rentals are subject to certain regulations.
                Per the city's "Short Term Residential Rentals" ordinance, hosts
                they must register their units with the city and pay a 14% hotel tax.
                Hosts are limited to renting their primary residence for a maximum of 90 days per calendar year,
                unless they have obtained a conditional use permit. Violations of these regulations may result in fines and penalties.
                In addition to the regulations in San Francisco, California has additional regulations that affect Airbnb rentals.
                For example, state law requires all hosts to provide certain information to guests,
                such as emergency contact information and information on smoke detectors and fire extinguishers""")
    st.markdown('Partha Mehta code')
    # create the code that gives us the table based on the laws that affect
    illegal_rentings = listings.groupby(['host_id', 'host_name', 'maximum_nights']).size().reset_index(
        name='illegal_rooms')
    illegal_rentings = illegal_rentings.sort_values(by=['maximum_nights'], ascending=False)
    illegal_rentings = illegal_rentings[illegal_rentings['maximum_nights'] >= 90]
    st.write(illegal_rentings)
    url5 = "https://sfplanning.org/"
    st.markdown("[Sfplanning](%s)" % url5)
    url6 = "https://www.lodgify.com/guides/short-term-rental-rules-california/"
    st.markdown("[Lodgify](%s)" % url6)

# plot a word map
with tab_plots:
    st.subheader('*San ​​Francisco Word Map*')
    text = ' '.join([text for text in listings['name']])
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110, background_color='black').generate(
        str(text))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.image(wordcloud.to_image(), caption='Wordcloud')

# ------------------------------------------------- ------------------------------FOURTH TAB: CONCLUSIONS ---------------- -------------------------------------------------- ------------------
tab_plots = tabs[3]
with tab_plots:
    image = Image.open('sunsetwide.png')
    st.header('Conclusions')
    st.image(image, width=800)
    st.write('Golden Gate Park: Image created with DALL-E-2')
    st.subheader('*EDA San Francisco Conclusions*')
    st.markdown('- Airbnb is a popular vacation rental platform in San Francisco.')
    st.markdown('- According to Inside Airbnb data, in 2021 there were over 12,000 active listings in the city.')
    st.markdown(
        '- Pacific Heights is the most expensive neighborhood for Airbnb rentals in San Francisco, known for its luxury homes and gorgeous views of the Golden Gate Bridge and the Bay.')
    st.markdown(
        '- Other expensive neighborhoods for San Francisco Airbnb rentals include: Russian Hill, Presidio Heights, Presidio')
    st.markdown(
        '- According to Inside Airbnb data, in 2021 the Bayview neighborhood is the cheapest for Airbnb rentals in San Francisco')
    st.markdown(
        '- Staying in neighborhoods like Bayview or Excelsior, may offer a better experience for tourists, given the local atmosphere')
    st.markdown(
        '- The city has a relatively high rate of property crime, with incidents of theft and burglary being especially common.')
    st.markdown('- Violent crime rates in San Francisco are relatively low compared to other major cities.')
    st.markdown(
        '- Although since 2019, San Francisco has a law that enforces an annual limit of 90 days of rental, we found Airbnb that possibly do not respect it.')
    st.markdown(
        '- The median salary for a data analyst in San Francisco, according to Glassdoor is around 90,000 a year. However, salaries can range from around 65,000 a year to over 130,000 a year')

# ------------------------------------------------- ------------------------------FIFTH TAB: LINEAR REGRESSION--------------- -------------------------------------------------- -------------------

# create a variable to encode these columns as they are categorical
columns_to_encode = ['listing_url', 'last_scraped', 'source', 'name', 'description', 'picture_url', 'host_url',
                     'host_name', 'host_since', 'host_location', 'host_response_time', 'host_response_rate',
                     'host_acceptance_rate',
                     'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_verifications',
                     'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'property_type',
                     'room_type', 'bathrooms_text',
                     'amenities', 'has_availability', 'calendar_last_scraped', 'first_review', 'last_review',
                     'instant_bookable', 'districts']

# create a for loop that iterates through all the columns and encodes them
for column in columns_to_encode:
    encode = preprocessing.LabelEncoder()
    encode.fit(listings[column])
    listings[column] = encode.transform(listings[column])
listings.sort_values(by='price', ascending=True, inplace=True)

# fit the linear regression model and split it into training set data and a test set by doing a train_test_split
l_reg = LinearRegression()
X = listings[
    ['id', 'neighbourhood_cleansed', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews',
     'reviews_per_month', 'calculated_host_listings_count', 'availability_365']]
y = listings['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
l_reg.fit(X_train, y_train)

# We calculate the different performance measures of the model such as Mean Squared Error, R2 Score, Mean Absolute Error, Root Mean Squared Error.
predicts = l_reg.predict(X_test)
print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
print("R2 Score: ", r2_score(y_test, predicts) * 100)
print("Mean Absolute Error: ", mean_absolute_error(y_test, predicts))
print("Root Mean Squared Error: ", mean_squared_error(y_test, predicts))

# create a dataframe to compare the current values ​​and the predicted values
lr_pred_df = pd.DataFrame({
    # create a pandas dataframe from two arrays: one containing the actual test values ​​and one containing the predicted values ​​for the tests. The data frame is then used to display the first 20 rows.
    'actual_values': np.array(y_test).flatten(),
    'predicted_values': predicts.flatten()}).head(20)

tab_plots = tabs[4]

# we create a function to compare the datasets that are the same when downloading these
with tab_plots:
    st.header('Linear regression: Prices')
    st.subheader('*Linear regression based on Dwi Gustin Nurdialit*')
    url6 = "https://medium.com/analytics-vidhya/python-exploratory-data-analysis-eda-on-nyc-airbnb-cbeabd622e30"
    st.markdown("[Online access](%s)" % url6)
    st.markdown(
        'We are going to use this mathematical model to explain the relationship between several independent variables and several dependent variables')
    st.markdown('*With this we will try to predict the value of the price based on the other independent variables*')

# look for the object-type columns to encode them
with tab_plots:
    st.markdown('We look for the object type columns to encode them')
    code = """
searching_dtype = listings.dtypes == object
list(listings.loc[:,searching_dtype])
"""
    st.code(code)

# encode the categorical variables
with tab_plots:
    st.markdown('We encode the categorical variables with a for loop that iterates over them')
    code = """
    # create a variable to encode these columns as they are categorical
    columns_to_encode = ['listing_url', 'last_scraped', 'source', 'name', 'description', 'picture_url', 'host_url', 'host_name', 'host_since', 'host_location', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                     'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed', 'property_type', 'room_type', 'bathrooms_text',
                     'amenities', 'has_availability', 'calendar_last_scraped', 'first_review', 'last_review', 'instant_bookable', 'districts']
    # create a for loop that iterates through all the columns and encodes them
    for column in columns_to_encode:
    encode = preprocessing.LabelEncoder()
    encode.fit(listings[column])
    listings[column] = encode.transform(listings[column])
    listings.sort_values(by='price',ascending=True,inplace=True)    
"""
    st.code(code, language='python')

# fit the linear regression model and split it into training set data and a test set by doing a train_test_split
with tab_plots:
    st.markdown(
        'We fit the linear regression model and split it into training set data and a test set by doing a train_test_split')
    code = """
    l_reg = LinearRegression()
    X = listings[['id','neighbourhood_cleansed','latitude','longitude','room_type','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']]
    y = listings['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    l_reg.fit(X_train,y_train)
"""
    st.code(code, language='python')

# We calculate the different performance measures of the model such as Mean Squared Error, R2 Score, Mean Absolute Error, Root Mean Squared Error.
with tab_plots:
    st.markdown(
        'We calculate the different performance measures of the model such as Mean Squared Error, R2 Score, Mean Absolute Error, Root Mean Squared Error.')
    code = """
    predicts = l_reg.predict(X_test)
    print("Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(y_test, predicts)))
    print("R2 Score: ", r2_score(y_test,predicts)*100)
    print("Mean Absolute Error: ", mean_absolute_error(y_test,predicts))
    print("Root Mean Squared Error: ", mean_squared_error(y_test,predicts))
"""
    st.code(code, language='python')
    st.markdown("""**Mean Squared Error (MSE): A measure of the difference between the predicted values ​​and the current values. A low MSE value indicates that the model is making accurate predictions.**
- The MSE is 95.91673833503985 is a very high value, so it indicates a bad prediction of the data, since there is a big difference between the prices.
**R2 Score: It is a measure of how well the model is able to explain the variation in the data. A value close to 1 indicates good model performance, while a value close to 0 indicates poor performance.**
- The R2 Score explains only 2.4139739201838206% of the variation in the data. This suggests that the model does not have a good fit to the data and is not capable of accurately predicting the target values ​​from the input values.
**Mean Absolute Error (MAE): It is a measure of the difference between the predicted values ​​and the actual values. It is the average of the magnitude of the differences between the predicted values ​​and the actual values.**
- The MAE is 75.38753534057028 indicates that the model has a mean error of 75.38753534057028 units in its predictions. A model with a very high MAE is not necessarily a bad model, but it indicates that the model's predictions are less accurate.
**Root Mean Squared Error (MSE): It is a measure of the difference between the predicted values ​​and the actual values. A low MSE value indicates that the model is making accurate predictions.**
- A SERM value of 9200.020692832502 indicates that the model has a root mean square error of 9200.020692832502 units in its predictions. It is a measure of the accuracy of the predictions. The smaller the MSRE value, the better the accuracy of the model.""")

# create a dataframe to compare the current values ​​and the predicted values
with tab_plots:
    st.markdown(
        'We create a dataframe to compare the current values ​​and the predicted values ​​and two arrays with the actual and predicted values')
    code = """
    # create a dataframe to compare the current values ​​and the predicted values
    lr_pred_df = pd.DataFrame({
    #create a pandas dataframe from two arrays: one containing the actual test values ​​and one containing the predicted values ​​for the tests. The data frame is then used to display the first 20 rows.
        'actual_values': np.array(y_test).flatten(),
        'predicted_values': predicts.flatten()}).head(20)"""
    st.code(code, language='python')

with tab_plots:
    # calculate the slope and y-intercept of the regression line
    slope, y_intercept = np.polyfit(lr_pred_df.actual_values, lr_pred_df.predicted_values, 1)
    # calculate the values ​​of x and y for the linear regression
    reg_x = lr_pred_df.actual_values
    reg_y = slope * reg_x + y_intercept

    # We plot a scatter with dispersion points and the regression line, seeing the price values ​​that it predicts for us
    fig = go.Figure(data=[
        go.Scatter(x=lr_pred_df.actual_values, y=lr_pred_df.predicted_values, mode='markers',
                   name='Scatter Plot', marker=dict(color='red')),
        go.Scatter(x=reg_x, y=reg_y, mode='lines', name='Regression Line', marker=dict(color='green'))
    ])

    fig.update_layout(
        xaxis_title='Current Prices',
        yaxis_title='Prices Prediction',
        template='plotly_dark',
    )
    st.subheader('*Linear Regression: Current Prices Vs Prediction Prices*')
    st.plotly_chart(fig)
    st.markdown(
        '**CONCLUSION: It may be necessary to tune the model or use a different technique to improve the accuracy of the predictions**')
    st.markdown(
        '**Also the data type used may not be the best for this type of model or even for doing a linear regression**')

# ----------------------------------------------------------------------------------END----------------------------------------------------------------------------------------------------------------