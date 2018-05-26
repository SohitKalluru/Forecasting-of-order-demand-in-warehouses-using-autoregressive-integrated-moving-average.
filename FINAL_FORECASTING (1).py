
# coding: utf-8

# # Forecasting of order demand in warehouses using autoregressive integrated moving average 

# ## Importing packages

# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from statsmodels.tsa.stattools import adfuller
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}

pylab.rcParams.update(params)
DataFrame=pd.read_csv('Historical Product Demand.csv')
DataFrame['Pandas_Datestamp'] = pd.to_datetime(DataFrame['Date'], infer_datetime_format=True)
DataFrame['Year'] = pd.DatetimeIndex(DataFrame['Date']).year
DataFrame['Month'] = pd.DatetimeIndex(DataFrame['Date']).month
DataFrame.sort_values(by='Pandas_Datestamp')
DataFrame.dropna()
#CHANGING STRING TO NUMERIC
DataFrame.Order_Demand = pd.to_numeric(DataFrame['Order_Demand'], errors='coerce')
Warehouse = DataFrame.Warehouse.unique() #GIVES 4 UNQUE WH
print DataFrame


# In[2]:


print len(Warehouse)


# In[3]:


print Warehouse


# # Data visualization 
# 

# In[11]:


from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
for i in range(0, len(Warehouse)):
    WH_S=pd.DataFrame(DataFrame[DataFrame['Warehouse']== Warehouse[i]])
    WH_S_2012=WH_S[WH_S['Year']==2012]
    WH_S_2012=pd.DataFrame(WH_S.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2012= WH_S_2012.sort_values('Order_Demand', ascending=False)
    WH_S_2013=WH_S[WH_S['Year']==2013]
    WH_S_2013=pd.DataFrame(WH_S_2013.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2013=WH_S_2013.sort_values('Order_Demand', ascending=False)
    WH_S_2014=WH_S[WH_S['Year']==2014]
    WH_S_2014=pd.DataFrame(WH_S_2014.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2014=WH_S_2014.sort_values('Order_Demand', ascending=False)
    WH_S_2015=WH_S[WH_S['Year']==2015]
    WH_S_2015=pd.DataFrame(WH_S_2015.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2015=WH_S_2015.sort_values('Order_Demand', ascending=False)
    WH_S_2016=WH_S[WH_S['Year']==2016]
    WH_S_2016=pd.DataFrame(WH_S_2016.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2016=WH_S_2016.sort_values('Order_Demand', ascending=False)
    trace1 = go.Bar(x=WH_S_2012['Product_Category'],  y=WH_S_2012['Order_Demand'], name='Year_2012')
    trace2 = go.Bar(x=WH_S_2013['Product_Category'],  y=WH_S_2013['Order_Demand'], name='Year_2013')
    trace3 = go.Bar(x=WH_S_2014['Product_Category'],  y=WH_S_2014['Order_Demand'], name='Year_2014')
    trace4 = go.Bar(x=WH_S_2015['Product_Category'],  y=WH_S_2015['Order_Demand'], name='Year_2015')
    trace5 = go.Bar(x=WH_S_2016['Product_Category'],  y=WH_S_2016['Order_Demand'], name='Year_2016')
    fig = tools.make_subplots(rows=2, cols=5)
    fig.append_trace(trace5, 1, 1)
    fig.append_trace(trace4, 1, 2)
    fig.append_trace(trace3, 1, 3)
    fig.append_trace(trace2, 1, 4)
    fig.append_trace(trace1, 1, 5)
    layout=fig['layout'].update(height=500, width=1200, title='Order demand vs product category with respect to all years for '+ str (Warehouse[i]),xaxis=dict(
        title='Product Category',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Order Demand',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f')))
    py.iplot(fig, filename='stacked-subplots', layout=layout)
    plot(fig, filename='stacked-subplots')


# ## Function to separate warehouse according to year and concatenate it vertically

# In[12]:


#FUNCTION FOR DIFFERENT WAREHOUSES
def diff_warehouse(Whse_A):
  
    #SCALING THE ORDER DEMAND
    
    WH_A=pd.DataFrame(DataFrame[DataFrame['Warehouse']== Whse_A]) #EXTRACTING A SPECIFIC WAREHOUSE
    cols_to_norm = ['Order_Demand'] #SCALING
    WH_A[cols_to_norm] = WH_A[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min())) #SCALING
    WH_A.sort_values(by='Pandas_Datestamp')


    #SEPARATING AS PER YEAR
    WH_A_2012=WH_A[WH_A['Year']==2012]
    WH_A_2012['Month'] = WH_A_2012['Pandas_Datestamp'].apply(lambda x: x.strftime('%Y-%m-01')) #FIRST DAY OF EVERY MONTH 
    WH_A_2012=pd.DataFrame(WH_A_2012.groupby('Month', as_index=False)['Order_Demand'].mean())
    WH_A_2012.sort_values(by='Month')

    WH_A_2013=WH_A[WH_A['Year']==2013]
    WH_A_2013['Month'] = WH_A_2013['Pandas_Datestamp'].apply(lambda x: x.strftime('%Y-%m-01')) #FIRST DAY OF EVERY MONTH
    WH_A_2013=pd.DataFrame(WH_A_2013.groupby('Month', as_index=False)['Order_Demand'].mean())
    WH_A_2013.sort_values(by='Month')

    WH_A_2014=WH_A[WH_A['Year']==2014]
    WH_A_2014['Month'] = WH_A_2014['Pandas_Datestamp'].apply(lambda x: x.strftime('%Y-%m-01')) #FIRST DAY OF EVERY MONTH
    WH_A_2014=pd.DataFrame(WH_A_2014.groupby('Month', as_index=False)['Order_Demand'].mean())
    WH_A_2014.sort_values(by='Month')

    WH_A_2015=WH_A[WH_A['Year']==2015]
    WH_A_2015['Month'] = WH_A_2015['Pandas_Datestamp'].apply(lambda x: x.strftime('%Y-%m-01')) #FIRST DAY OF EVERY MONTH
    WH_A_2015=pd.DataFrame(WH_A_2015.groupby('Month', as_index=False)['Order_Demand'].mean())
    WH_A_2015.sort_values(by='Month')

    WH_A_2016=WH_A[WH_A['Year']==2016]
    WH_A_2016['Month'] = WH_A_2016['Pandas_Datestamp'].apply(lambda x: x.strftime('%Y-%m-01')) #FIRST DAY OF EVERY MONTH
    WH_A_2016=pd.DataFrame(WH_A_2016.groupby('Month', as_index=False)['Order_Demand'].mean())
    WH_A_2016.sort_values(by='Month')

#CONCATENATION

    WH_A_ALLYEARS = pd.concat([WH_A_2012,WH_A_2013,WH_A_2014,WH_A_2015,WH_A_2016]).reset_index(drop=True) #, axis=1
    WH_A_ALLYEARS.index = WH_A_ALLYEARS['Month']
    WH_A_ALLYEARS.drop(columns='Month')
    WH_A_ALLYEARS= WH_A_ALLYEARS.drop(columns='Month')
    WH_A_ALLYEARS.reset_index(inplace=True)
    WH_A_ALLYEARS['Month'] = pd.to_datetime(WH_A_ALLYEARS['Month'])
    WH_A_ALLYEARS = WH_A_ALLYEARS.set_index('Month')
    #ROLLING AVERGAGE FORMULA - TRIAL WITH MOVING WINDOW
    WH_A_ALLYEARS['MA_3']= WH_A_ALLYEARS.Order_Demand.rolling(3).mean()
    WH_A_ALLYEARS['MA_3_std']= WH_A_ALLYEARS.Order_Demand.rolling(3).std() #QUATERLY
    WH_A_ALLYEARS['Warehouse'] = Whse_A
    return WH_A_ALLYEARS


# In[13]:


class color:
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
for i in range(0,len(Warehouse)):
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\t\t\t %s \n'% Warehouse[i] + color.END
    print diff_warehouse(Warehouse[i])
    


# In[14]:


def Plot_Original(WH_A_ALLYEARS):
    Actual1 = go.Scatter(x=WH_A_ALLYEARS.index, y=WH_A_ALLYEARS.Order_Demand, mode = 'lines+markers',name = 'Actual')
    MA_3 = go.Scatter(x=WH_A_ALLYEARS.index, y=WH_A_ALLYEARS.MA_3, mode = 'lines+markers',name = '3-PERIOD MOVING AVERAGE')
    MA_3_std = go.Scatter(x=WH_A_ALLYEARS.index, y=WH_A_ALLYEARS.MA_3_std, mode = 'lines+markers',name = '3-PERIOD MOVING STANDARD DEVIATION')
    data1= [Actual1, MA_3, MA_3_std]
    layout = go.Layout(
    title='Order Demand for ' + Warehouse[i],
    xaxis=dict(
        title='Years',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Order Demand',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
    plot_2 = go.Figure(data=data1, layout=layout)
   # plot_image = py.plot(plot_2, filename='styling-names')
    return plot(plot_2, filename='styling-names')


# In[15]:


for i in range(0,len(Warehouse)):
    print Plot_Original(diff_warehouse(Warehouse[i]))


# ## Function to test the stationarity of the graph

# In[16]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


# In[17]:


class color:
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
for i in range(0,len(Warehouse)):
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    print test_stationarity(diff_warehouse(Warehouse[i]).Order_Demand)
    
    


# In[18]:


for i in range(0,len(Warehouse)):
    decomp=sm.tsa.seasonal_decompose(diff_warehouse(Warehouse[i]).Order_Demand,freq = 12)
    plt.show()
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    decomp.plot()
    


# ## `for` loop for creating ACF and PACF plots

# In[19]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

for i in range(0,len(Warehouse)):
    plot_acf(diff_warehouse(Warehouse[i]).Order_Demand)
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    plt.show()
    plot_pacf(diff_warehouse(Warehouse[i]).Order_Demand)
    plt.show()


# ## Method 2 - Auto Arima

# In[20]:


from pyramid.arima import auto_arima
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot

for i in range(0,len(Warehouse)):
    train = diff_warehouse(Warehouse[i]).iloc[0:int(len(diff_warehouse(Warehouse[i]))*0.7)]
    test = diff_warehouse(Warehouse[i]).iloc[int(len(diff_warehouse(Warehouse[i]))*0.7)+1:]
    print  '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    stepwise_model = auto_arima(train.Order_Demand, start_p=1, start_q=1,max_p=3, max_q=3, m=25,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,)
    order_in = stepwise_model.order
    seasonal_order_in = stepwise_model.seasonal_order
    print'Least AIC ',stepwise_model.aic()
    print'Least BIC ',stepwise_model.bic()
   
    mod = sm.tsa.statespace.SARIMAX(train.Order_Demand, trend='n', order=order_in , seasonal_order=seasonal_order_in,enforce_invertibility=False)
    results = mod.fit()
    print '\n\n\n',results.summary()
    
    print '\n\n\n\t\t\t\t\t\t Plotting Diagnostics'
    results.plot_diagnostics(figsize=(20, 14))
    plt.show()
    
    print '\n\n\n\t\t\t\t\t\t Forecasting using trained model - 70% Data '
    prediction_1 = results.get_forecast('2017-12')
    prediction_1_ci = prediction_1.conf_int()
    print(prediction_1.predicted_mean['2016-01':'2017-10'])
    
    pred = prediction_1.predicted_mean['2016-01':'2017-10']
    
    print '\n\n\n\t\t\t\t\t\t Dataframe of Forecasting '
    Prediction_df = pd.DataFrame(pred,columns=['ORDER_DEMAND_FORECAST'])
    print Prediction_df
    
    #print("%s",% Warehouse[i].center())
    
   # Prediction_df = pd.DataFrame(prediction_1,columns=['ORDER_DEMAND_FORECAST'])
   # Prediction_df

    Given = go.Scatter(x=diff_warehouse(Warehouse[i]).index, y=diff_warehouse(Warehouse[i]).Order_Demand, mode = 'lines+markers',name = 'Order_Demand'+ Warehouse[i])
    Predicted=go.Scatter(x=Prediction_df.index, y=Prediction_df.ORDER_DEMAND_FORECAST, mode = 'lines+markers',name = 'Predicted_Order_Demand'+ Warehouse[i])
    #Actual = go.Scatter(x=WH_A_ALLYEARS_FO.index, y=WH_A_ALLYEARS_FO.Order_Demand, mode = 'lines+markers',name = 'Actual')
    Final_Visu =[Given,Predicted]
    layout = go.Layout(
    title='Forecasted Order Demand for ' + Warehouse[i],
    xaxis=dict(
        title='Years',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Order Demand',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
    Visu = go.Figure(data=Final_Visu, layout=layout)
    print plot(Visu, filename='styling-names')


# In[21]:


a = diff_warehouse(Warehouse[1])
print a


# In[22]:


#DIFFERENCING
#DataFrame.diff(periods=1, axis=0)
#Differencing is selected based on the Dickey Fuller test
a['DIFFERENCING']=diff_warehouse(Warehouse[1])['Order_Demand'].diff(periods=1) 
a['DIFFERENCING'].head()
len(a)
a
#PERIODS CAN CHANGE - WE NEED CHECK WHICH ORDER OF DIFFERENCING IS BETTER. FIRST ORDER REMOVES TREND, SECOND ORDER REMOVES SEASONALITY


# ## Dividing the data into testing data and training data

# In[23]:


train1 = a.iloc[0:40]
train1.head()
len(train1)
train1


# In[24]:


test_stationarity(train1.DIFFERENCING.dropna(inplace=False))


# In[25]:


train1.reset_index(inplace=True)
train1['Month'] = pd.to_datetime(train1['Month'])
train1 = train1.set_index('Month')
train1.DIFFERENCING.dropna(inplace=True)


# In[28]:


decomp=sm.tsa.seasonal_decompose(train1.DIFFERENCING,freq = 12)
decomp.plot()
plt.show()


# In[29]:


from pyramid.arima import auto_arima
stepwise_model = auto_arima(train1.Order_Demand, start_p=1, start_q=1,max_p=3, max_q=3, m=25,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,)
print(stepwise_model.bic())


# In[30]:


mod = sm.tsa.statespace.SARIMAX(train1.Order_Demand, trend='n', order=(1, 1, 0),seasonal_order=(0, 1, 0, 25),enforce_invertibility=False)
results = mod.fit()
print results.summary()


# In[31]:


results.plot_diagnostics(figsize=(20, 14))
plt.show()


# In[32]:


train1


# In[33]:


pred_1 = results.get_forecast('2017-12')
pred2_ci = pred_1.conf_int()
print(pred_1.predicted_mean['2016-12':'2017-10'])


# In[34]:


pred = pred_1.predicted_mean['2016-12':'2017-10']
pred


# In[35]:


PE_1 = pd.DataFrame(pred,columns=['ORDER_DEMAND_FORECAST'])
PE_1


# In[36]:


Given1 = go.Scatter(x=diff_warehouse(Warehouse[1]).index, y=diff_warehouse(Warehouse[1]).Order_Demand, mode = 'lines+markers',name = 'Order_Demand'+ Warehouse[1])
Predicted1=go.Scatter(x=PE_1.index, y=PE_1.ORDER_DEMAND_FORECAST, mode = 'lines+markers',name = 'Predicted_Order_Demand'+ Warehouse[1])
#Actual = go.Scatter(x=WH_A_ALLYEARS_FO.index, y=WH_A_ALLYEARS_FO.Order_Demand, mode = 'lines+markers',name = 'Actual')
Final_Visu1 =[Given1,Predicted1]
plot(Final_Visu1)


# In[37]:


#a=a.drop(columns='MA_3')
#a=a.drop(columns='MA_3_std')
#a = a.drop(columns='DIFFERENCING')
#a = a.drop(columns='Warehouse')
a


# ## Final forecast visualization code

# In[236]:


train = diff_warehouse(Warehouse[3]).iloc[0:int(len(diff_warehouse(Warehouse[3]))*0.7)]
test = diff_warehouse(Warehouse[3]).iloc[int(len(diff_warehouse(Warehouse[3]))*0.7)+1:]
print  '\n\n\n\n___________________________________________________________________________________________________________________________'
print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
stepwise_model = auto_arima(train.Order_Demand, start_p=1, start_q=1,max_p=3, max_q=3, m=25,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,)
order_in = stepwise_model.order
seasonal_order_in = stepwise_model.seasonal_order
print'Least AIC ',stepwise_model.aic()
print'Least BIC ',stepwise_model.bic()
   
mod = sm.tsa.statespace.SARIMAX(train.Order_Demand, trend='n', order=order_in , seasonal_order=seasonal_order_in,enforce_invertibility=False)
results = mod.fit()
print '\n\n\n',results.summary()
    
print '\n\n\n\t\t\t\t\t\t Plotting Diagnostics'
results.plot_diagnostics(figsize=(20, 14))
plt.show()
    
print '\n\n\n\t\t\t\t\t\t Forecasting using trained model - 70% Data '
prediction_1 = results.get_forecast('2017-12')
prediction_1_ci = prediction_1.conf_int()
print(prediction_1.predicted_mean['2016-01':'2017-10'])
    
pred = prediction_1.predicted_mean['2016-01':'2017-10']
    
print '\n\n\n\t\t\t\t\t\t Dataframe of Forecasting '
Prediction_df = pd.DataFrame(pred,columns=['ORDER_DEMAND_FORECAST'])
print Prediction_df
    
    #print("%s",% Warehouse[i].center())
    
   # Prediction_df = pd.DataFrame(prediction_1,columns=['ORDER_DEMAND_FORECAST'])
   # Prediction_df

Given = go.Scatter(x=diff_warehouse(Warehouse[]).index, y=diff_warehouse(Warehouse[i]).Order_Demand, mode = 'lines+markers',name = 'Order_Demand'+ Warehouse[i])
Predicted=go.Scatter(x=Prediction_df.index, y=Prediction_df.ORDER_DEMAND_FORECAST, mode = 'lines+markers',name = 'Predicted_Order_Demand'+ Warehouse[i])
Actual = go.Scatter(x=WH_A_ALLYEARS_FO.index, y=WH_A_ALLYEARS_FO.Order_Demand, mode = 'lines+markers',name = 'Actual')
Final_Visu =[Given,Predicted]
layout = go.Layout(
title='Forecasted Order Demand for ' + Warehouse[i],
xaxis=dict(
        title='Years',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
yaxis=dict(
        title='Order Demand',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
Visu = go.Figure(data=Final_Visu, layout=layout)
print plot(Visu, filename='styling-names')



# In[38]:


#diff_warehouse(Warehouse[3])=diff_warehouse(Warehouse[3]).drop(columns=['MA_3', 'MA_3_std'])
xyz=diff_warehouse(Warehouse[3])
xyz = xyz.drop(columns=['MA_3', 'MA_3_std'])
xyz


# ## One more style of visualization including the band of forecast

# In[238]:


# plot time series and out of sample prediction

ax = xyz['2012':].plot(label='Observed', color='#006699')
prediction_1.predicted_mean.plot(ax=ax, label='Out-of-Sample Forecast', color='#ff0066')
ax.fill_between(prediction_1_ci .index,
                prediction_1_ci .iloc[:, 0],
                prediction_1_ci .iloc[:, 1], color='#ff0066', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2016-12'), PE_1.index[-1], alpha=.15, zorder=-1, color='grey')
ax.set_xlabel('Date')
ax.set_ylabel('Order Demand')
#plt.legend()
#plt.savefig('./img/out_of_sample_pred.png')
plt.show()

