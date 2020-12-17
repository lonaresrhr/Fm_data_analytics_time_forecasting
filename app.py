import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import load_model
#from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



st.beta_set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False) ### To hide matplotlib.pyplot error have to correct later on


image = Image.open('logo1.jpeg')

st.image(image, width = 250)
st.title(' Fm data Analytics App')

col1 = st.sidebar
col2, col3 = st.beta_columns((3,1))

col3.title('Analytics Section')




#col1.header('upload input data file')
uploaded_file = col1.file_uploader("Upload your input  csv file",type=["csv"])

@st.cache(allow_output_mutation=True)
def upload_function():
	data1 = st.cache(pd.read_csv)(uploaded_file,parse_dates=True,index_col='Timestamp')
	data2=st.cache(pd.read_csv)(uploaded_file,parse_dates=True)
	try:
		datax=data1
		datax['Datetime'] = pd.to_datetime(data2.Timestamp ,format='%Y-%m-%d %H:%M') 
		data1=data1
		
	except:
		data1=st.cache(pd.read_csv)(uploaded_file,parse_dates=True,index_col='Timestamp',dayfirst=True)
		#data6['Datetime'] = pd.to_datetime(data2.Timestamp ,format='%d-%m-%Y %H:%M') 
	
	return data1,data2
	
@st.cache(allow_output_mutation=True)
def upload_function_1():	
	data1 =st.cache(pd.read_csv)('Effimax - Sunidhi - History.csv',parse_dates=True,index_col='Timestamp',dayfirst=True)
	data1=data1.loc[data1['EFF_Boiler_ON'] == 1]
	data2=st.cache(pd.read_csv)('Effimax - Sunidhi - History.csv')
	data2=data2.loc[data2['EFF_Boiler_ON'] == 1]
	
	return data1,data2




################### Uploading user selectec files  ###################################

if uploaded_file is not None:

	
	data1,data2=upload_function()
	
	
	
else :
	data1,data2=upload_function_1()

orignaldata=data1


################# Creating Multiselection features sidebars #################################
    
feature= col1.multiselect("select The feature for data analytics",orignaldata.columns)
plot_timeline =col1.radio('Plot data Timeline', ['Minute-Wise','Hourly', 'Daily', 'Weekly', 'Monthly','Weekend'])



####################   Displaying   datafrmes  ###################################
is_check = col3.checkbox("Display orignal Data")
if is_check:
    col2.write("Orignal Data")
    col2.write(orignaldata)
    



data=pd.DataFrame(orignaldata[feature])
data=data.dropna()

is_check1=col3.checkbox("Display selected feature Data")
if is_check1:
	col2.write("Selected Feature Data")
    	#col2.write(data)
	hourly = data.resample('H').mean() 
	hourly=hourly.dropna()
	daily = data.resample('D').mean() 
	daily=daily.dropna()
	weekly = data.resample('W').mean() 
	weekly=weekly.dropna()
	monthly = data.resample('M').mean()
	monthly=monthly.dropna()
	minitly=data.resample('min').mean() 
	minitly=minitly.dropna()
    	

	

	if plot_timeline == 'Minute-Wise':
		col2.write(minitly)
	
	if plot_timeline == 'Hourly':
		col2.write(hourly)
	
	if plot_timeline == 'Weekly':
		col2.write(weekly)
	
	if plot_timeline == 'Daily':
		col2.write(daily)
	
	if plot_timeline == 'Monthly':
		col2.write(monthly)

####################    Plotting correlation Matrix  ##################################
l=len(feature)
list1=range(0,l)

is_check5=col3.checkbox("Display selected features correlation matrix with all features")

data3=data2.drop(columns=['Timestamp'])
corr = data3.corr()

if is_check5:
	for i in list1:
		#data1[feature[i]].plot.hist()
		corr1=corr[feature[i]].sort_values(ascending=False)
		df=pd.DataFrame(corr1)
		col2.write(df)
		y=[]
		z=[]
		r=corr1.size
		#print(r)
		X=range(0,r)
		for j in X:
    		#print(corr1[i])
    			y.append(corr1[j])
    			z.append(corr1.index[j])
		plt.figure(figsize=(15,10))
		sns.barplot(y,z) 
		
		plt.show()
		plt.title(feature[i]+"_Correlation_Matrix")
		col2.pyplot()
     
#####################	Plotting Histograms	#############################################

is_check4=col3.checkbox("Display selected feature histograms")

if is_check4:
	for i in list1:
		data1[feature[i]].plot.hist()
		#data1[feature[i]].hist()
		plt.show()
		plt.legend([feature[i]])
		col2.pyplot()

####################	 Ploting hourly and daily and weekly available data for selected feature ###############################
   




is_check2=col3.checkbox("Display selected feature data timeline plots")

if is_check2:
	hourly = data.resample('H').mean() 
	hourly=hourly.dropna()
	# Converting to daily mean 
	daily = data.resample('D').mean() 
	daily=daily.dropna()
	# Converting to weekly mean 
	weekly = data.resample('W').mean() 
	weekly=weekly.dropna()
	# Converting to monthly mean 
	monthly = data.resample('M').mean()
	monthly=monthly.dropna()
	#Converting to minitly mean
	minitly=data.resample('min').mean() 
	minitly=minitly.dropna()

	if plot_timeline == 'Minute-Wise':
		col2.line_chart(minitly)
	
	if plot_timeline == 'Hourly':
		col2.line_chart(hourly)
	
	if plot_timeline == 'Weekly':
		col2.line_chart(weekly)
	
	if plot_timeline == 'Daily':
		col2.line_chart(daily)
	
	if plot_timeline == 'Monthly':
		col2.line_chart(monthly)


############################	Plotting Mean timeline data bar chart     ############################
data6=data2
try:
	data6['Datetime'] = pd.to_datetime(data2.Timestamp ,format='%Y-%m-%d %H:%M') 
except:
	data6['Datetime'] = pd.to_datetime(data2.Timestamp ,format='%d-%m-%Y %H:%M') 
i=data6


i['year']=i.Datetime.dt.year 
i['month']=i.Datetime.dt.month 
i['day']=i.Datetime.dt.day
i['Hour']=i.Datetime.dt.hour 
i['Minute']=i.Datetime.dt.minute
i["Week"]=i.Datetime.dt.day_name()
data4=i


data4['day of week']=data4['Datetime'].dt.dayofweek 
temp = data4['Datetime']
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 
temp2 = data4['Datetime'].apply(applyer) 
data4['weekend']=temp2
data4.index = data4['Datetime'] # indexing the Datetime to get the time period on the x-axis. 

is_check3=col3.checkbox("Display selected feature Mean value timeline bar plots")


if is_check3:
	for i in list1:
		
   
		if plot_timeline == 'Minute-Wise':
			col2.bar_chart(data4.groupby('Minute')[feature[i]].mean())
	
		if plot_timeline == 'Hourly':
			#st.line_chart(data4.groupby('Hour')[feature[i]].mean())
	
			col2.bar_chart(data4.groupby('Hour')[feature[i]].mean())
	
		if plot_timeline == 'Weekly':
			col2.bar_chart(data4.groupby('Week')[feature[i]].mean())
	
		if plot_timeline == 'Daily':
			col2.bar_chart(data4.groupby('day')[feature[i]].mean())
	
		if plot_timeline == 'Monthly':
			col2.bar_chart(data4.groupby('month')[feature[i]].mean())
		if plot_timeline == 'Weekend':
			col2.bar_chart(data4.groupby('weekend')[feature[i]].mean())


#####################  SEF Calculation  ####################################

is_check_sef = col3.checkbox("SEF")

#################### Time series prediction ################################
scaler = MinMaxScaler()
def f1(v,model):
    t=[[v]]
    X=scaler.fit_transform(t)
    X1=np.reshape(X, (X.shape[0], 1, X.shape[1]))
    p=model.predict(X1)
    p1=scaler.inverse_transform(p)
    return p1
#y=data1.EFF_Efficiency[-1]
#print(y)
#y=6.789
#result=f1(y)
#result




columns_time_forecasting=["EFF_Efficiency",'EFF_Enthalpy_Loss']

feature_t= col1.multiselect("Select The feature for time forecasting",columns_time_forecasting)
#plot_timeline1 = col1.radio('selct the predition for next', ['Minute','Hour', 'Day', 'Week', 'Month'])
plot_timeline1 = col1.radio('selct the predition for next', ['Hour', 'Day', 'Week', 'Month'])
l_t=len(feature_t)
list_t=range(0,l_t)

data_t=pd.DataFrame(orignaldata[feature_t])
data_t=data_t.dropna()
is_check_Tsp = col3.checkbox("Time Series prediction")
if is_check_Tsp:
	hourly = data_t.resample('H').mean() 
	hourly=hourly.dropna()
		# Converting to daily mean 
	daily = data_t.resample('D').mean() 
	daily=daily.dropna()
		# Converting to weekly mean 
	weekly = data_t.resample('W').mean() 
	weekly=weekly.dropna()
		# Converting to monthly mean 
	monthly = data_t.resample('M').mean()
	monthly=monthly.dropna()
		#Converting to minitly mean
	minitly=data_t.resample('min').mean() 
	minitly=minitly.dropna()

	for j in list_t:
		
   
		if plot_timeline1 == 'Minute':
			model = load_model(feature_t[j]+'/'+feature_t[j]+'_hourly'+'.h5')
			result=f1(y,model)
			col2.line_chart(minitly[feature_t[j]])
	
		if plot_timeline1 == 'Hour':
			
			model = load_model('Time_forecast_features/'+feature_t[j]+'/'+feature_t[j]+'_hourly'+'.h5')
			#model = load_model(feature_t[j]+'/'+feature_t[j]+'_hourly'+'.h5')
			#model = load_model(feature_t[j]+'_hourly'+'.h5')
			#model = load_model('EFF_efficiency/'+feature_t[j]+'_hourly'+'.h5')
			y=hourly[feature_t[j]][-1]
			result=f1(y,model)
			col2.write(hourly[feature_t[j]])
			col2.write("predicted "+feature_t[j]+" for next one hour= ")
			col2.write(result)
	
			
	
			#col2.line_chart(hourly[feature_t[j]])
	
		if plot_timeline1 == 'Week':
			#model = load_model(feature_t[j]+'/'+feature_t[j]+'_weekly'+'.h5')
			model = load_model(feature_t[j]+'_weekly'+'.h5')
			y=weekly[feature_t[j]][-1]
			result=f1(y,model)
			col2.write(weekly[feature_t[j]])
			col2.write("predicted "+feature_t[j]+" for next one week= ")
			col2.write(result)
	
			#col2.line_chart(weekly[feature_t[j]])
	
		if plot_timeline1 == 'Day':
			#model = load_model(feature_t[j]+'/'+feature_t[j]+'_daily'+'.h5')
			model = load_model(feature_t[j]+'_daily'+'.h5')
			#model = load_model('EFF_Efficiency_daily.h5')
			y=daily[feature_t[j]][-1]
			result=f1(y,model)
			col2.write(daily[feature_t[j]])
			col2.write("predicted "+feature_t[j]+"value for next one day = ")
			col2.write(result)
	
			
			#col2.line_chart(daily[feature_t[j]])
			#col2.write("3#######")
	
		if plot_timeline1 == 'Month':
			#col2.line_chart(monthly[feature_t[j]])
			col2.write("Not sufficient data for Monthly Prediction")
		
		

#################### Extra features ########################################
is_check_Ef_1 = col3.checkbox("....")
is_check_Ef_2 = col3.checkbox(".....")


