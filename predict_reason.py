import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# importing the data
data_set = pd.read_csv("mediacompany.csv")


# removing the junk column from dataset
data_set = data_set.drop('Unnamed: 7',axis = 1)


# change the format of date column
data_set['Date'] = pd.to_datetime(data_set['Date']).dt.date


# creating a new deriving column 
from datetime import date

d0 = date(2017, 2, 28)
d1 = data_set.Date
dif = (d1 - d0).dt.days
data_set['Day'] = dif


# Days vs Views_show graph
data_set.plot.line(x='Day', y='Views_show')


# Scatter plot of Days vs Views_show
colors = (0,0,0)
area = np.pi*3
plt.scatter(data_set.Day, data_set.Views_show, s=area, c=colors, alpha=0.5)
plt.title('scatter plot of DayvsViews_show')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# plot for days vs Views_show and days vs Ad_impressions
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel("Day")
host.set_ylabel("View_Show")
par1.set_ylabel("Ad_impression")

color1 = plt.cm.viridis(0.2)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(data_set.Day, data_set.Views_show, color=color1, label="View_Show")
p2, = par1.plot(data_set.Day, data_set.Ad_impression, color=color2, label="Ad_impression")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

par2.spines['right'].set_position(('outward', 60))
par2.xaxis.set_ticks([])

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

plt.savefig("multiple_y-axis.png",bbox_inches='tight')


# Another derive column for weekdays 
data_set['Weekday'] = (data_set['Day']+3)%7
data_set.Weekday.replace(0,7, inplace=True)
data_set['Weekday'] = data_set['Weekday'].astype(int)
data_set.head()


# Running first model Weekday & visitors
X = data_set[['Visitors' , 'Weekday']]
Y = data_set['Views_show']

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,Y)

import statsmodels.api as sm
X = sm.add_constant(X)

lm_1 = sm.OLS(Y,X).fit()
print(lm_1.summary())


# Creating a weekend variable with value 1 for weekends and 0 for other weekdays
def cond(i):
    if i % 7 == 5:
        return 1
    elif i % 7 == 4:
        return 0
    else:
        return 0
    return i

data_set['Weekend']=[cond(i) for i in data_set['Day']]


# Running second model Weekend & visitors
X = data_set[['Visitors','Weekend']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_2 = sm.OLS(Y,X).fit()
print(lm_2.summary())


# Running Third model Weekend, visitors & Character_A
X = data_set[['Visitors','Weekend','Character_A']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_3 = sm.OLS(Y,X).fit()
print(lm_3.summary())


# Creating a new variable from Views_show
data_set['last_view'] = np.roll(data_set['Views_show'], 1)
data_set.last_view.replace(108961,0, inplace=True)


# Running fourth model Weekend, visitors , Character_A, last_view
X = data_set[['Visitors' , 'Weekend','Character_A', 'last_view']]
Y = data_set['Views_show']


import statsmodels.api as sm
X = sm.add_constant(X)

lm_4 = sm.OLS(Y,X).fit()
print(lm_4.summary())

    
# Running fifth model Weekdend, Character_A & views_platform
X = data_set[['Weekend' , 'Character_A', 'Views_platform']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_5 = sm.OLS(Y,X).fit()
print(lm_5.summary())


# Running Third model Weekend, visitors & Character_A
X = data_set[['Weekend','Character_A', 'Visitors']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_6 = sm.OLS(Y,X).fit()
print(lm_6.summary())


# Running Third model Weekend, visitors & Character_A
X = data_set[['Weekend','Character_A', 'Visitors', 'Ad_impression']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_7 = sm.OLS(Y,X).fit()
print(lm_7.summary())


# Running Third model Weekend, visitors & Character_A
X = data_set[['Weekend','Character_A', 'Ad_impression']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_8 = sm.OLS(Y,X).fit()
print(lm_8.summary())


# Ad_impression in million
data_set['Ad_impression_million'] = data_set['Ad_impression']/1000000


# Running Third model Weekend, visitors & Character_A
X = data_set[['Weekend','Character_A', 'Ad_impression_million','Cricket_match_india']]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_9 = sm.OLS(Y,X).fit()
print(lm_9.summary())


# Running Third model Weekend, visitors & Character_A
X = data_set[['Weekend','Character_A', 'Ad_impression_million',]]
Y = data_set['Views_show']

import statsmodels.api as sm
X = sm.add_constant(X)

lm_10 = sm.OLS(Y,X).fit()
print(lm_10.summary())


# Making prediction using the lm_10 model
X = data_set[['Weekend','Character_A','Ad_impression_million']]
X = sm.add_constant(X)
Predicted_views = lm_10.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(data_set.Views_show, Predicted_views)
r_squared = r2_score(data_set.Views_show, Predicted_views)

print('Mean_Squared_Error:' ,mse)
print('r_square_value:' ,r_squared)


# Actual vs Predicted
c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,data_set.Views_show, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="red", linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Views', fontsize=16)


# Error terms
c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,data_set.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error terms', fontsize=20)
plt.xlabel('Index', fontsize=18)
plt.ylabel('Views_show-Predicted_views', fontsize=16)















