# Forecasting of order demand in warehouses using autoregressive integrated moving average.

Authors:  **VIJAY GANESH SRINIVASAN**, **SOHIT REDDY KALLURU**, **RAMAKRISHNA POLEPEDDI**.

YouTube Video:  [Link](https://www.youtube.com/watch?v=Yf_kFP6B_rQ)

---

**NOTE**:  For more details on the statistical understanding of the project kindly read *Introduction to time series and forecasting
Book by Peter J. Brockwell and Richard A. Davis, Production and Operations Analysis by Steven Nahmias and Supply Chain Engineering: Models and Applications by A. Ravi Ravindran, Donald P. Warsing, Jr.* 

---
## FORECASTING
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Forecasting_Title_Image.PNG)

---
## Project outline
- The objective of the project is to forecast the order demand using AUTOREGRESSIVE INTEGRATED MOVING AVERAGE model for 4 warehouses respectively.
- For this analysis we have downloaded the data from Kaggle. (https://www.kaggle.com/felixzhao/productdemandforecasting/data)
- The basics of ACF, PACF, rolling mean average, rolling standard deviation and correlogram are explained in this documentation.
- By the end of the documentation you'll have a clear idea about **A**utoregressive **I**ntegrated **M**oving **A**verage or **ARIMA** model, data visualization, data analysis, statistical library functions in python and creation of interactive plots using plotly.  

---

## What makes the dataset interesting
- The dataset contains historical product demand for a manufacturing company with footprints globally. 
- The company provides thousands of products within dozens of product categories for 7 years. There are four central warehouses to ship products within the region it is responsible for.
- The data is available in the .csv format which allows us to perform the dataframe operations easily.

---

### NOTE : Packages to install before running this program

### Plotly - Modern Visualization for the Data Era

- It is one important package to be installed to have interactive plots. It is very easy to use.

### Installing instructions

- To install Plotly's python package, use the package manager pip inside your terminal.

```
$ pip install plotly 
```
- After installing plotly run python and configure plotly by entering your credentials.

```
import plotly
plotly.tools.set_credentials_file(username='MyAccount', api_key='********')
```
- Use [this hyperlink](https://plot.ly/feed) to create  an account and to generate API key follow the instructions mentioned in the website. 

---

## 5 Steps towards Forecasting

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/5_Steps_Towards_Forecasting.PNG)

---

## Introduction to ARIMA

---

- ARIMA is a forecasting technique. ARIMA– Auto Regressive Integrated Moving Average the key tool in Time Series Analysis.
- Models that relate the present value of a series to past values and past prediction errors - these are called ARIMA models.
- ARIMA models provide an approach to time series forecasting. 
- ARIMA is a forecasting technique that projects the future values of a series based entirely on its own inertia.
- Exponential smoothing and ARIMA models are the two most widely-used approaches to time series forecasting. 
- Exponential smoothing models are based on a description of trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.
- Its main application is in the area of short term forecasting requiring at least 40 historical data points. 
- It works best when your data exhibits a stable or consistent pattern over time with a minimum amount of outliers.
- ARIMA is usually superior to exponential smoothing techniques when the data is reasonably long and the correlation between past observations is stable.

--- 

## Program explanation

---

### Code summary

---

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Steps_ARIMA_FORECASTING.PNG)

- The flow of the program is excuted in 2 ways.
- One is as per the flow chart and the alternative one is by using Auto Arima algorithm which is pre-installed package in Anaconda Python.
- The uploaded code is excuted in the Jupyter environment. 
- Kindly as mentioned above install the packages required to run the program.

---

### Packages to import in Python before running the program

```
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
import statsmodels.api as sm
import matplotlib.pylab as pylab
```
- To set the figure size for all the plots together you can use the following code

```
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}

pylab.rcParams.update(params)
```
- To perform time series operations easily we are changing the normal date format from the csv file to Pandas Datestamp.

```
DataFrame=pd.read_csv('Historical Product Demand.csv')
DataFrame['Pandas_Datestamp'] = pd.to_datetime(DataFrame['Date'], infer_datetime_format=True)
DataFrame['Year'] = pd.DatetimeIndex(DataFrame['Date']).year
DataFrame['Month'] = pd.DatetimeIndex(DataFrame['Date']).month
DataFrame.sort_values(by='Pandas_Datestamp')
```
- In the Pandas world 'obj' data type is nothing but string. This is one of the challenges we faced. `pd.to_numeric` is the command used to change the column of dataframe to numeric type where we could perform mathematical operations.

```
#CHANGING STRING TO NUMERIC
DataFrame.Order_Demand = pd.to_numeric(DataFrame['Order_Demand'], errors='coerce')
```
### Data visualization

- The below code runs in a `for` loop where for all the 4 warehouse the graphs are plotted using plotly interactive plots.
- This is one of the easy way to use the facility of `sub plots` provided by the plotly. 
- **Only the snippet of code is given. Kindly refer .ipynb file or .py file for elaborate explanation** 

```
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
for i in range(0, len(Warehouse)):
    WH_S=pd.DataFrame(DataFrame[DataFrame['Warehouse']== Warehouse[i]])
    WH_S_2012=WH_S[WH_S['Year']==2012]
    WH_S_2012=pd.DataFrame(WH_S.groupby('Product_Category', as_index=False)['Order_Demand'].mean())
    WH_S_2012= WH_S_2012.sort_values('Order_Demand', ascending=False)
    trace1 = go.Bar(x=WH_S_2012['Product_Category'],  y=WH_S_2012['Order_Demand'], name='Year_2012')
    trace2 = go.Bar(x=WH_S_2013['Product_Category'],  y=WH_S_2013['Order_Demand'], name='Year_2013')
    trace3 = go.Bar(x=WH_S_2014['Product_Category'],  y=WH_S_2014['Order_Demand'], name='Year_2014')
    fig = tools.make_subplots(rows=2, cols=5)
    fig.append_trace(trace3, 1, 3)
    fig.append_trace(trace2, 1, 4)
    fig.append_trace(trace1, 1, 5)
    fig['layout'].update(height=500, width=1200, title='Order demand vs product category with respect to all years for '+ str (Warehouse[i]),xaxis=dict(
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

```

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/WH_A.png)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/WH_J.png)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/WH_S.png)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/WH_C.png)

---

### Writing functions for different Warehouses

- The function is written to consolidate all the warehouses demand and order demand.

```
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
    
    #CONCATENATION

    WH_A_ALLYEARS = pd.concat([WH_A_2012,WH_A_2013,WH_A_2014,WH_A_2015,WH_A_2016]).reset_index(drop=True) #, axis=1
    WH_A_ALLYEARS.index = WH_A_ALLYEARS['Month']
    WH_A_ALLYEARS.drop(columns='Month')
    WH_A_ALLYEARS= WH_A_ALLYEARS.drop(columns='Month')
    WH_A_ALLYEARS.reset_index(inplace=True)
    WH_A_ALLYEARS['Month'] = pd.to_datetime(WH_A_ALLYEARS['Month'])
    WH_A_ALLYEARS = WH_A_ALLYEARS.set_index('Month')
    
  ```
  - In the next half of the function the moving average is calculated. ARIMA methodology also allows models to be built that incorporate both autoregressive and moving average parameters together. So that we calculate moving average too, or in python langauge we calculate rolling average for all the years in the same function.
  
  ```
   #ROLLING AVERGAGE FORMULA - TRIAL WITH MOVING WINDOW
    WH_A_ALLYEARS['MA_3']= WH_A_ALLYEARS.Order_Demand.rolling(3).mean()
    WH_A_ALLYEARS['MA_3_std']= WH_A_ALLYEARS.Order_Demand.rolling(3).std() #QUATERLY
    WH_A_ALLYEARS['Warehouse'] = Whse_A
    return WH_A_ALLYEARS
   ```
   ### Graphical representation of each and every warehouse demand in time series model
   
   - To get a clear idea on the time series model we plot all the warehouse details in the based on the years and order demand.
   - There are lots of functions written in this project to avoid the number of steps and computation time.
   
   ```
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
   for i in range(0,len(Warehouse)):
    print Plot_Original(diff_warehouse(Warehouse[i])) 
   ```


![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Order_Demand_Whse_A.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Order_Demand_Whse_J.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Order_Demand_Whse_S.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Order_Demand_Whse_C.PNG)

### Dickey - Fuller test statistic
- This is one of the statistical tests for checking stationarity. 
- Here the null hypothesis is that the time series is non-stationary. 
- The test results comprise of a test statistic and some critical values for difference confidence levels. 
- If the ‘test statistic’ is less than the ‘critical value’, we can reject the null hypothesis and say that the series is stationary.

```
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
    
    class color:
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
for i in range(0,len(Warehouse)):
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    print test_stationarity(diff_warehouse(Warehouse[i]).Order_Demand)
    
 ```
---
Dickey-Fuller test for A warehouse
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/DF_A.PNG)
---
Dickey-Fuller test for J warehouse
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/DF_J.PNG)
---
Dickey-Fuller test for C warehouse
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/DF_C.PNG)
---
Dickey-Fuller test for S warehouse
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/DF_S.PNG)
---

- From the Dickey-Fuller test we can say that for warehouse S, there is trend and seasonality since the p-value is greater than 0.05 (the test statistic is more than the critical value). So we have to use the decompose function from the statistics library to perform this operation.
- Below shown is the decomposition of a actual value graph. Time series decomposition works by splitting a time series into three components: seasonality, trends and random fluctiation.

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/General_Explanation.PNG)


**Above image courtesy: https://anomaly.io/seasonal-trend-decomposition-in-r/**



![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/WH_S_Seasonality.PNG)
   
### Significance of ACF and PACF

- The `statsmodels` were used to get the ACF and PACF plots.
- Below is the program coded in `for` loop so that it could generate ACF and PACF plots for all the warehouses.

```
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

for i in range(0,len(Warehouse)):
    plot_acf(diff_warehouse(Warehouse[i]).Order_Demand)
    print '\n\n\n\n___________________________________________________________________________________________________________________________'
    print color.BOLD  + '\n\n\t\t\t\t\t\t\t %s \n'% Warehouse[i] + color.END
    plt.show()
    plot_pacf(diff_warehouse(Warehouse[i]).Order_Demand)
    plt.show()
```




**Autocorrelation Function (ACF)**
- It is a measure of the correlation between the time series with a lagged version of itself. 
- q – The lag value where the ACF chart crosses the upper confidence interval for the first time.
- Below the ACF plot for warehouse S


![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/ACF_S.PNG)


**Partial Autocorrelation Function (PACF)** 
- This measures the correlation between the time series with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. 
- p – The lag value where the PACF chart crosses the upper confidence interval for the first time.
- Below the PACF plot for warehouse S



![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/PACF_S.PNG)

- Till this method, we have done the manual coding that involves each and every step of ARIMA modeling.
- For reducing the steps and building the model easiily there is one algorithm called `auto arima`.
- In the next section we will be seeing how `auto arima` and its functions work.
- In the below model the data is separated into training data and testing data.
- In this training data is used to train the model for forecasting and testing data is used to test the data against the build model.

```
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
```
- p= order of the autoregressive part;
- d= degree of first differencing involved;
- q= order of the moving average part.
- In the Auto Arima function (p, d, q) is the non-seasonal part of the model, (P, D, Q) is the seasonal part of the model, m is the number of periods per season.
- We select these values based on AIC and BIC of the model. 

**AIC**

- The Akaike Information Critera (AIC) is a widely used measure of a statistical model. It basically quantifies the goodness of fit, the simplicity/parsimony, of the model into a single statistic.
- When comparing two models, the one with the lower AIC is generally “better”
- AIC= 2k-2ln(L) where, k corresponds to the number of estimated parameters in the model and L refers to the maximum value of the likelihood function for the model.
- If the values of AIC and BIC are minimum then we select those values of (p, d, q) (P, D, Q)m.

**Seasonal Autoregressive Integrated Moving Average**

```
mod = sm.tsa.statespace.SARIMAX(train.Order_Demand, trend='n', order=order_in , seasonal_order=seasonal_order_in,enforce_invertibility=False)
results = mod.fit()
print '\n\n\n',results.summary()
```

- Fits ARIMA models (including improved diagnostics) in a short command. It can also be used to perform regression with autocorrelated errors. This is a front end to arima() with a different back door.


### Plotting diagnostics


![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/Diagnostic_S.PNG)


- In the top right plot, we see that the red KDE line follows closely with the N(0,1) line (where N(0,1)) is the standard notation for a normal distribution with mean 0 and standard deviation of 1). This is a good indication that the residuals are normally distributed.
- The qq-plot on the bottom left shows that the ordered distribution of residuals (blue dots) does follows the linear trend of the samples taken from a standard normal distribution with N(0, 1)

## Final Forecasting

```
    print '\n\n\n\t\t\t\t\t\t Forecasting using trained model - 70% Data '
    prediction_1 = results.get_forecast('2017-12')
    prediction_1_ci = prediction_1.conf_int()
    print(prediction_1.predicted_mean['2016-01':'2017-10'])
    
    pred = prediction_1.predicted_mean['2016-01':'2017-10']
 ```
 - In the above snippet 70% train model is used to test and forecast more than 30% of data. 
 - The forecasting graphs are attached for the reference. 
    
```   
    print '\n\n\n\t\t\t\t\t\t Dataframe of Forecasting '
    Prediction_df = pd.DataFrame(pred,columns=['ORDER_DEMAND_FORECAST'])
    print Prediction_df
```
- The predictions results obtained were converted into dataframe using the above code.

```
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
```
- The above snippet is a `for` loop construction used to display all the forecasts of warehouses after a single block of code. 

## Results from analysis

![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/FC_WA_A.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/FC_WA_C.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/FC_WA_J.PNG)
![Image of Plot](https://github.com/IE-555/final-project-arima_forecasting_team/blob/master/images/FC_S_1.PNG)

---

## References

---

*The references are given in the structure of program.*
- Kindly [watch this Youtube video](https://www.youtube.com/watch?v=Aw77aMLj9uM) to know how ARIMA works.
- For primary data analysis [click here](https://www.bigskyassociates.com/blog/bid/372186/The-Data-Analysis-Process-5-Steps-To-Better-Decision-Making)
- On the information of how to clean the data [click here](http://chi2innovations.com/blog/discover-data-blog-series/how-clean-your-data-quickly-5-steps/)
- To learn about basic data pre-processing [click here](http://iasri.res.in/ebook/win_school_aa/notes/Data_Preprocessing.pdf)
- To learn the statistical concepts of time series and forecasting [click here](https://www.researchgate.net/file.PostFileLoader.html?id=55502f915f7f71d7a68b45df&assetKey=AS%3A273774321045510%401442284292820)
- To know about selecting particular data for rows and columns from pandas dataframe [click here](https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas)
- Why to go with [regression model](https://dss.princeton.edu/online_help/analysis/regression_intro.htm)
- What is [linear regression](https://www.statisticallysignificantconsulting.com/RegressionAnalysis.htm)
- To know [how to plot graph between 2 columns](https://stackoverflow.com/questions/17812978/how-to-plot-two-columns-of-a-pandas-data-frame-using-points)
- To know [how to change the pandas object to numeric](https://stackoverflow.com/questions/40095712/when-to-applypd-to-numeric-and-when-to-astypenp-float64-in-python)
- To learn about time series plots from plotly [click here](https://plot.ly/python/time-series/)
- To learn about normalizing one column in a dataframe [click here](https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe)
- Click to know [how to drop Nan objects in pandas dataframe](https://stackoverflow.com/questions/36370839/better-way-to-drop-nan-rows-in-pandas)
- To know about `groupby` and finding mean [click here](https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas)
- **To know how to use offline plots in plotly [click here](https://stackoverflow.com/questions/35315726/visualize-plotly-charts-in-spyder)**
- To learn about moving average [click here](https://www.investopedia.com/ask/answers/013015/what-are-main-advantages-and-disadvantages-using-simple-moving-average-sma.asp)

---

## How to Run the Code

---

*In this tutorial we are running the codes in Jupyter environment*

1. As mentioned in the very beginning step kindly register in plotly for username and API key as per the instructions given in the website and in our documentation.

2. Ensure that you have installed necessary Python packages. (Most of the packages except plotly are pre-installed packages. Kindly verify whether you have those packages)

3. Download the .ipynb file and store it in the Jupyter working directory.

4. Download the dataset in .csv format from the link provided and store it in the same directory where .ipynb file is stored.

5. You can run each and every cell separately in the python notebook file and obtain the output or you can run the whole python program using `run` command in Jupyter.

`run FINAL_FORECASTING.py`

---

## Why this project is good

---

- When it comes to ARIMA forecasting not all material available on internet is for Python. Many are for R. This documentation is covers the whole concept of ARIMA starting from the basics to the advanced level for Python environment.
- References are provided in such a way that there won't be any need for you to refer any other links or websites other than this documentation
- Nearly we deal with more than 1 million data in Pandas dataframe efficiently using the functions built for certain tasks.
- Last but not least we have provided a way to save the plots offline.

---

## Suggestions

---

- Since the dataset includes 4 warehouse locations, if there are location coordinates provided for warehouse we can plot them in the map and predict the demand based on the location.
- This could be extended to the suppliers and customers involved with the 4 warehouses and their product demands. If customer-A wants to buy a product or stock a product, based on the existing forecast data, location we can provide which warehouse would be better option.
- This could be further extended to Q,R model to predict the re-order point and EOQ for the inventory storage.
- If we have the customer data we can build ABC or continous review model. 
