import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import load_model
#from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
from io import StringIO


#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Fm data Analytics App',layout='wide')
#st.set_option('deprecation.showPyplotGlobalUse', False) ### To hide matplotlib.pyplot error have to correct later on


image = Image.open('logo1.jpeg')

st.image(image, width = 250)
#st.title(' Fm data Analytics App')

col1 = st.sidebar
col2, col3 = st.beta_columns((3,1))

col3.title('Analytics Section')



with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
  

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
	
#@st.cache(allow_output_mutation=True)
def upload_function_1():	
	data1 = pd.read_csv('VardhmanBudhni_Yarns_ Gas_combine.csv')
	data2=pd.read_csv('VardhmanBudhni_Yarns_ Gas_combine.csv')
	try:
		
		data1['Datetime'] = pd.to_datetime(data1.Timestamp ,format='%Y-%m-%d %H:%M:%S') 
		data1.drop('Timestamp', axis=1, inplace=True)
		data1.set_index("Datetime", inplace = True)
 
		
		
	except:
		data1['Datetime'] = pd.to_datetime(data1.Timestamp ,format='%d-%m-%Y %H:%M:%S') 
		data1.drop('Timestamp', axis=1, inplace=True)
		data1.set_index("Datetime", inplace = True)
	
	return data1,data2
###########################################################################################

#global train_upload
#train_upload = st.file_uploader("Upload csv data", type=['csv'])
#if (train_upload is not None):
 #   train_features = train_upload.read()
 #   train_features = str(train_features,'utf-8')
 #   train_features = StringIO(train_features)
 #   train_features = pd.read_csv(train_features)
  #  st.write(train_features)


################### Uploading user selectec files  ###################################

if uploaded_file is not None:

	data1 = pd.read_csv(uploaded_file)
	#data2 =  pd.read_csv(uploaded_file)
	data2=data1
	try:
		
		data1['Datetime'] = pd.to_datetime(data1.Timestamp ,format='%Y-%m-%d %H:%M:%S') 
		data1.drop('Timestamp', axis=1, inplace=True)
		data1.set_index("Datetime", inplace = True)
 
		
		
	except:
		data1['Datetime'] = pd.to_datetime(data1.Timestamp ,format='%d-%m-%Y %H:%M:%S') 
		data1.drop('Timestamp', axis=1, inplace=True)
		data1.set_index("Datetime", inplace = True)
 
	
	
	
else :
	data1,data2=upload_function_1()

orignaldata=data1
	

################# Creating Multiselection features sidebars #################################
    
feature= col1.multiselect("Select the features for data analytics",orignaldata.columns)
plot_timeline =col1.radio('Plot data Timeline', ['Daily','Hourly','Minute-Wise', 'Weekly/Weekdays', 'Monthly','Weekend'])



####################   Displaying   datafrmes  ###################################
is_check = col3.checkbox("Display orignal Data")
if is_check:
    col2.write("Orignal Data")
    col2.write(orignaldata)
    



data=pd.DataFrame(orignaldata[feature])
#data=data.dropna()

is_check1=col3.checkbox("Display selected features Data")
if is_check1:
    col2.write("Selected Feature Data")
    col2.write(data)

####################    Plotting correlation Matrix  ##################################
l=len(feature)
list1=range(0,l)

is_check5=col3.checkbox("Display selected features correlation matrix with all features")

#data3=data2
#corr = data3.corr()
corr=orignaldata.corr()
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
		if (l<=5) :
			f1=plt.figure(figsize=(10,3))
		elif(l<=10):
			f1=plt.figure(figsize=(10,5))
		else:
			f1=plt.figure(figsize=(15,10))
		
			
		
		sns.barplot(y,z) 
		
		plt.show()
		plt.title(feature[i]+"_Correlation_Matrix")
		col2.pyplot(f1)
     
#####################	Plotting Histograms	#############################################

is_check4=col3.checkbox("Display selected features histograms")

if is_check4:
	for i in list1:
		f2=plt.figure(figsize=(10,5))
		data1[feature[i]].plot.hist()
		plt.show()
		plt.legend([feature[i]])
		col2.pyplot(f2)


####################	 Ploting hourly and daily and weekly available data for selected feature ###############################
   


data=data1

is_check2=col3.checkbox("Display selected features data timeline plots")

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
	
	if plot_timeline == 'Weekly/Weekdays':
		col2.line_chart(weekly)
	
	if plot_timeline == 'Daily':
		col2.line_chart(daily)
	
	if plot_timeline == 'Monthly':
		col2.line_chart(monthly)
	if plot_timeline == 'Weekend':
		col2.write("Weekend option is not applicable for displaying data timeline plots")

############################	Plotting Mean timeline data bar chart     ############################
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0 

data6=data2


	
	

is_check3=col3.checkbox("Display selected features Mean value timeline bar plots")

if is_check3:
	i=data1
	i.reset_index(inplace=True)
	i['year']=i.Datetime.dt.year 
	i['month']=i.Datetime.dt.month 
	i['day']=i.Datetime.dt.day
	i['Hour']=i.Datetime.dt.hour 
	i['Minute']=i.Datetime.dt.minute
	i["Week"]=i.Datetime.dt.day_name()
	data4=i


	data4['day of week']=data4['Datetime'].dt.dayofweek 
	temp = data4['Datetime']

	temp2 = data4['Datetime'].apply(applyer) 
	data4['weekend']=temp2
	data4.index = data4['Datetime'] # indexing the Datetime to get the time period on the x-axis. 
	#data4=data_fun()

	if plot_timeline == 'Minute-Wise':
		bar_data=data4.groupby('Minute')[feature].mean()
	
	elif plot_timeline == 'Hourly':
			#st.line_chart(data4.groupby('Hour')[feature[i]].mean())
	
		bar_data=data4.groupby('Hour')[feature].mean()
	
	elif plot_timeline == 'Weekly/Weekdays':
		bar_data=data4.groupby('Week')[feature].mean()
	
	elif plot_timeline == 'Daily':
		bar_data=data4.groupby('day')[feature].mean()
	
	elif plot_timeline == 'Monthly':
		bar_data=data4.groupby('month')[feature].mean()
	elif plot_timeline == 'Weekend':
		bar_data=data4.groupby('weekend')[feature].mean()
	col2.bar_chart(bar_data)

	

#####################  SEF Calculation  ####################################

is_check_sef = col3.checkbox("SEF")

#################### Time series prediction ################################
scaler = MinMaxScaler()
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




#columns_time_forecasting=["EFF_Efficiency",'EFF_Enthalpy_Loss']

#feature_t= col1.multiselect("Select the features for time forecasting",columns_time_forecasting)
#plot_timeline1 = col1.radio('selct the predition for next', ['Minute','Hour', 'Day', 'Week', 'Month'])
#@st.cache(allow_output_mutation=True)






#@st.cache(suppress_st_warning=True) 
def split_data(data1):
	train_size = int(len(data1) * 0.8)
	print(train_size)
	train = data1[0:train_size]
	test=data1[train_size:len(data1)]
	valid=data1[train_size:len(data1)]
	return train,test,valid

#@st.cache(allow_output_mutation=True)
#@st.cache(suppress_st_warning=True) 
#@st.cache(suppress_st_warning=True) 
def plot_data(train,valid,feat):
	#f1=plt.plot(train.index, train, label='Train')
	#f1=plt.plot(valid.index,valid, label='Valid')
	#col2.line_chart(test)
	#f1=plt.figure(figsize=(10,5))
	#f1=plt.plot(train)
	#plt.plot(valid)
	#ax.legend(["Training data", "validation data"])
	#col2.line_chart(train)
	#col2.line_chart(valid)
	#plt.plot(train.index, train, label='Train')
	#plt.plot(valid.index,valid, label='Valid')
	#plt.show()
	
	ax = train.plot( kind = 'line',figsize=(15,4),title=feat)
	valid.plot(kind = 'line', ax=ax,figsize=(15,4))
	ax.legend(["Training data", "validation data"])
	plt.show()

	#ax.title(fet)
	col2.pyplot(plt)
	plt.show()
	ax.remove()
#@st.cache(suppress_st_warning=True) 
def plot_predections(train,valid,y_lstm,feat):
	ax = train.plot( kind = 'line',figsize=(15,4),title=feat)
	valid.plot(kind = 'line', ax=ax,figsize=(15,4))
	y_lstm["Forcasted_"+feat].plot(kind = 'line', ax=ax,figsize=(15,4))
	ax.legend(["Training data", "validation data","LSTM Forecast"])
	plt.show()

	#ax.title(fet)
	col2.pyplot(plt)
	plt.show()
	ax.remove()

	
#@st.cache(suppress_st_warning=True) 
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#@st.cache(suppress_st_warning=True) 
def data_process(train,test):
	# normalize the dataset
	
	# dataset=dataset.reshape(-1,1)
	train=pd.DataFrame(train)
	test=pd.DataFrame(test)
	train1=scaler.fit_transform(train)
	test1=scaler.fit_transform(test)
	#dataset = scaler.fit_transform(data)
	# reshape into X=t and Y=t+1
	look_back = 1
	trainX, trainY = create_dataset(train1, look_back)
	testX, testY = create_dataset(test1, look_back)
	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	return trainX, trainY,testX, testY

#@st.cache(allow_output_mutation=True)
def create_model(trainX, trainY,ep):
	# create and fit the LSTM network
	look_back=1
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	#if X=="daily":
	model.fit(trainX, trainY, epochs=ep, batch_size=1, verbose=2)
	return model

#@st.cache(suppress_st_warning=True) 
def Make_predictions(test,trainX, trainY,testX, testY,model,feat):
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)
	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY1 = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])	
	testPredict1=pd.DataFrame(testPredict)
	testPredict1.columns = ["Forcasted_"+feat]
	mean_test=np.mean(testPredict1)
	x=pd.DataFrame([mean_test,mean_test])
	testPredict2=testPredict1.append(x, ignore_index=True)
	y_lstm = test.copy()
	y_lstm=pd.DataFrame(y_lstm)
	y_lstm['DateTime'] = y_lstm.index
	y_lstm.reset_index(inplace=True)
	#del y_lstm['Timestamp']
	y_lstm = pd.concat([y_lstm,testPredict2],axis=1)
	y_lstm = y_lstm.set_index('DateTime')
	print(y_lstm)
	return y_lstm





plot_timeline1 = col1.radio('selct the predition ', ['Day', 'Hour', 'Week', 'Month'])
l_t=len(feature)
list_t=range(0,l_t)

data_t=pd.DataFrame(orignaldata[feature])
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
		
   
		
		if plot_timeline1 == 'Hour':
			train,test,valid=split_data(hourly[feature[j]])
			plot_data(train,valid,feature[j])
			trainX, trainY,testX, testY=data_process(train,test)
			model=create_model(trainX, trainY,ep=10)
			y_lstm=Make_predictions(test,trainX, trainY,testX, testY,model,feature[j])
			plot_predections(train,valid,y_lstm,feature[j])
			
			
	
			#col2.line_chart(hourly[feature_t[j]])
	
		if plot_timeline1 == 'Week':
			train,test,valid=split_data(weekly[feature[j]])
			plot_data(train,valid,feature[j])
			trainX, trainY,testX, testY=data_process(train,test)
			model=create_model(trainX, trainY,ep=100)
			y_lstm=Make_predictions(test,trainX, trainY,testX, testY,model,feature[j])
			plot_predections(train,valid,y_lstm,feature[j])
			
			#col2.line_chart(weekly[feature_t[j]])
	
		if plot_timeline1 == 'Day':
			train,test,valid=split_data(daily[feature[j]])
			plot_data(train,valid,feature[j])
			trainX, trainY,testX, testY=data_process(train,test)
			model=create_model(trainX, trainY,ep=30)
			y_lstm=Make_predictions(test,trainX, trainY,testX, testY,model,feature[j])
			plot_predections(train,valid,y_lstm,feature[j])
			

			
			
			
			#col2.line_chart(daily[feature_t[j]])
			#col2.write("3#######")
	
		if plot_timeline1 == 'Month':
			#col2.line_chart(monthly[feature_t[j]])
			col2.write("Not sufficient data for Monthly Prediction")
		

#################### Extra features ########################################
is_check_Ef_1 = col3.checkbox("....")
is_check_Ef_2 = col3.checkbox(".....")




	
	



