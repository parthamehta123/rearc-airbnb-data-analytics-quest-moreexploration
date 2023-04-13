**Exploratory Data Analysis of AirBnB listings in one of the major locations of California - San Francisco**

Data Source: http://insideairbnb.com/get-the-data.html

Using the [Inside Airbnb](http://insideairbnb.com/) listings data for San Francisco, I was able to understand popular trends and predict SF listing prices given certain characteristics. 

Python Packages used in the project:
- NumPy
- Pandas
- Scikit-learn
- mpl_toolkits
- matplotlib
- plotly
- Tensorflow and Keras
- Seaborn
- Folium
- XGBoost, GradientBoostRegressor, RandomForest, AdaBoostRegressor, BaggingRegressor, DecisionTree
 
This project is meant to give deeper understanding of the airbnb listing data, and introduce how powerful and convenient scikit-learn package's algorithm functions are.

**Overview of the data**

Exploring an Airbnb Dataset

Tables / Descriptions

To download these files, copy/paste the link below into the address bar. The input files are located on drive link here.

https://drive.google.com/drive/folders/1z9v2WSWmwLSVTFNf0vYaIy-q_8WQh081?usp=sharing

Alternatively, pd.read_csv(url) should work just fine.

File	Description

listings.csv.gz	Detailed Listings Data

calendar.csv.gz	Detailed Calendar Data

reviews.csv.gz	Detailed Review Data

listings.csv	Summary information and metrics for listings in San Francisco

reviews.csv	Summary Review data and Listing ID

neighborhoods.csv	Neighborhood list for geo filter. Sourced from city or open source GIS files.

neighbourhoods.geojson	GeoJSON file of neighborhoods of the city

## Final Findings Summarized after analyzing listings.csv/calendar.csv/reviews.csv:
I used the notebook explore_analysis_summary_listingsreviewscalendarfiles.ipynb. I used the various data cleaning, data summary and data visualization, geo spatial visualization techniques to analyze the AirBnB listings of San Francisco region and found the following summary:

It (explore_analysis_summary_listingsreviewscalendarfiles.ipynb) asks and answers 3 questions :

1)  What are the busiest months in San Francisco and the price variations of the same? This helps in understanding the options of housing and the prices for the same.
  
  - The busiest months are August,September and October which is Fall season in the city characterized by warm, pleasant climates. Price is generally higher than average in       those months with sharp dips from December to May which are generally very cold.

2)  What are the type of rooms available? How does availability and price differ with type of room?
  
  - Hotel room has highest availability and highest price. Next highest listings and prices are for Private room and Entire home/apt. Shared room has lowest availability and      lowest price.

3)  Which neighbourhood has most listings? How does availability and price vary with neighborhood? Is price related to number of listings in that neighbourhood? My blog post      on medium summarises the insights for a non-techinical audience.
  
  - Downtown,Mission, Western Addition and South of Market have most listings whereas Presidio, South of Market and Golden Gate Park have highest number of days available         (Note: Golden Gate Park has only 3 listings availability so availability is not an accurate metric) and Presidio, Seacliff and Twin Peaks are the most expensive.

## Final Findings Summarized after analyzing listings.csv.gz:
I used the notebook explore_analysis_detailed_listingsfile.ipynb and made below inferences for the following questions.

1) What factors are most strongly associated with the price of an Airbnb rental in San Francisco? For example, how does neighborhood affect pricing?

2) What neighborhoods in San Francisco have the highest demand for Airbnb rentals?

3) Can you identify any interesting trends or patterns in the demand for Airbnb rentals in San Francisco over time?

4) Can you predict the occupancy rate and/or price of a given Airbnb listing in San Francisco based on its attributes and availability?

- Most of the listings are clustered near the bart stations and center of the city / Most Airbnb rentals in San Francisco are in the north-east coastal areas.
- Prices in most neighbors are skewed to the right. The Downtown/Civic area, while having the lowest average price, shows a bimodal distribution. This suggests a gap between the rich and the poor.
- Superhosts, in general, provide better renting experience, their rating in all aspects were 0.02 - 0.2 higher in rating than regular hosts.
- Mission (421), Western Addition (306) and South of Market (291) at the top 3 neighborhoods with most listings
- Average price of all SF listings is `$`180.11.
- Prices vary wildly based on property and room types.
- Presidio (`$`301.83), Seacliff (`$`292.6), Twin Peaks (`$`246.34) are the most expensive neighborhoods.
- Majority of listings are rented for their entirety, although private room is a close second. This is the most important factor when people choose where to stay.
- Accomodates is the most important factor, meaning that most people who use Airbnb at SF travel in groups.
- Almost all of listings are Entire home/apt.
- Most frequent words in summaries show that more hosts talk about the surrounding area rather than the listing itself.
- Listings with prices around `$`100 - 300 get the most reviews, meaning that they are booked most often.
- We have used multiple machine learning algorithms to successfully make an estimation for Airbnb rent price. In which Gradient Boosting Regressor and Neural Network have the best performance.
- I also did few Ensemble Algorithms in this notebook if you scroll up just for experimentation.

- Noticed that there are large number of listings in San Francisco.
- Noticed that the price distribution is righ skewed with very few listings as costly as 25k per night.
- The mean price of listings with room type as Entire home / apt or Private room seem to be higher than Hotel & Shared room in San Francisco per night and this price is around $25 'and' $16 which is greater than average price of $1 for Shared / Hotel room. So, we can conclude from our exploration that room type does affect price for the listings.
- There are a lot of listings available for a minimum stay of 1, 2 days as well as 30 days. But, there are more listings for minimum stay options as 1/2 days compared to 30 days as law of SF is that, guests can't stay for a duration more than 2 weeks.
- Also one can notice that, majority of the listings are of Entire home and Private room type.
- Noticed that majority of lisitngs which are also highly reviewed listings are concentrated near the downtown and tourist regions of San Francisco. So, we can conclude that listing based on neighbourhood does affect price of the listing as well.
- Noticed that the most common words used in listing names are "private", "cozy", "beautiful", "home", "room", "spacious", "studio", "view" etc.

***Problem Statement:(From notebook Airbnb Data Exploration and Analysis For Recommendations.ipynb)***

Simplifying the SF Airbnb Search Process: The goal is to develop a model that allows customers to find suitable Airbnb’s in the city of San Francisco based on pre-selected features such as bedrooms, neighbourhood, price and more. While the Airbnb website offers a few filters to search with, our model goes beyond filtering, exploring similarities across neighbourhoods. Customers also have the option to narrow their search based on how our model clusters Airbnb’s!

***Project Context & Motivation:(From notebook Airbnb Data Exploration and Analysis For Recommendations.ipynb)***
By developing a recommendation model that suggests Airbnb’s as per personal choices, we can help various types of customers select Airbnb’s from myriads of options available in San Francisco that be-fit their preferences in a more efficient manner.
Such a recommendation tool and a user-interactive interface enables user-friendly service and attracts more customers to utilize it.

***Note***
I have also made a Streamlit application. You can refer the code in app.py file present in this repository.
To run the application, you can perform the below steps:
1. !rm ~/.streamlit/config.toml --> If you face a Toml config error, you should run this first, then reinstall streamlit using the below command 2nd or 3rd.
2. !pip install -q streamlit
3. !npm install localtunnel
4. !streamlit run /content/drive/MyDrive/airbnb-data-analysis/app.py &>/content/logs.txt &
5. !npx localtunnel --port 8501
6. !pip install pyngrok
7. !pip install streamlit
8. !pip install chart_studio

**Future work**
- These findings can be used for a comparative analysis with other nearby cities like Los Angeles.
- Sentiment analysis can be performed on the reviews.
- Further analysis of factors affecting price and ratings.
- We can also use Hypothesis Testing in the future by using A/B Testing for example. Using this, what we can do is, maybe see the profitability/revenue and more customer retention kind of KPI metrics by improving maybe on facility of the AirBnB rentals and give suggestions to people who are travelling due to some reason, check how these changes are impacting the growth of AirBnB and if the metrics are looking good, use that business model and maybe give recommendations to AirBnB or the people who are travelling. We can try getting confidence interval type of statistics to measure goodness of our model fit for the hypotheses.
- We can also use time series forecasting to detect all these metrics mentioned above.

***Medium Blog***
Two of my Medium Blogs on this topic are here:

- https://medium.com/@parthamehta10/this-is-how-i-estimate-price-for-airbnb-in-san-francisco-using-different-ml-algorithms-like-linear-5a1a8cc22ea7
- https://medium.com/@parthamehta10/3-insights-into-airbnb-in-san-francisco-c2c6bd6cd91d

***Tableau Workbook***
Download them from this link: https://github.com/parthamehta123/rearc-airbnb-data-analytics-quest-moreexploration/blob/master/Airbnb%20San%20Francisco%20Neighborhood%20Analysis.twbx
