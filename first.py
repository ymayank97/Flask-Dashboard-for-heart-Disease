import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from sklearn.externals import joblib
import traceback
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import json
from plotly import figure_factory as ff
from sklearn.metrics import roc_auc_score
from scipy import stats
app = Flask(__name__)

def data_category(dataset):
	dataset['Gender']=dataset['Gender'].replace([1,0], ['Male', 'Female'])
	dataset['Slope']=dataset['Slope'].replace([1,2,3], ['Upsloping','Flat','Down-sloping'])
	dataset['RestECG']=dataset['RestECG'].replace([0,1,2], ['Normal', 'Abnormality','Hypertrophy'])
	dataset['Exang']=dataset['Exang'].replace([1,0], ['Yes', 'No'])
	dataset['FBS']=dataset['FBS'].replace([1,0], ['Yes', 'No'])
	dataset['Thal']=dataset['Thal'].replace([3,6,7], ['Normal', 'Fixed Defect','Reversible defect'])
	dataset['CP']=dataset['CP'].replace([1,2,3,4], ['Typical angina', 'Atypical angina','Non-anginal pain','Asymptomatic pain'])
	dataset['Gender']=dataset['Gender'].astype('object')
	dataset['CP']=dataset['CP'].astype('object')
	dataset['Thal']=dataset['Thal'].astype('object')
	dataset['FBS']=dataset['FBS'].astype('object')
	dataset['Exang']=dataset['Exang'].astype('object')
	dataset['RestECG']=dataset['RestECG'].astype('object')
	dataset['Slope']=dataset['Slope'].astype('object')
	dataset['Goal']=dataset['Goal'].replace([1,2], ['Absence', 'Presence'])
	dataset['Goal']=dataset['Goal'].replace( ['Absence', 'Presence'],[0,1])
	return dataset

def data_category2(dataset):
	dataset['Gender']=dataset['Gender'].replace([1,0], ['Male', 'Female'])
	dataset['Slope']=dataset['Slope'].replace([1,2,3], ['Upsloping','Flat','Down-sloping'])
	dataset['RestECG']=dataset['RestECG'].replace([0,1,2], ['Normal', 'Abnormality','Hypertrophy'])
	dataset['Exang']=dataset['Exang'].replace([1,0], ['Yes', 'No'])
	dataset['FBS']=dataset['FBS'].replace([1,0], ['Yes', 'No'])
	dataset['Thal']=dataset['Thal'].replace([3,6,7], ['Normal', 'Fixed Defect','Reversible defect'])
	dataset['CP']=dataset['CP'].replace([1,2,3,4], ['Typical angina', 'Atypical angina','Non-anginal pain','Asymptomatic pain'])
	dataset['Gender']=dataset['Gender'].astype('object')
	dataset['CP']=dataset['CP'].astype('object')
	dataset['Thal']=dataset['Thal'].astype('object')
	dataset['FBS']=dataset['FBS'].astype('object')
	dataset['Exang']=dataset['Exang'].astype('object')
	dataset['RestECG']=dataset['RestECG'].astype('object')
	dataset['Slope']=dataset['Slope'].astype('object')
	return dataset
	
def load_data():
	# if request.method == 'POST':
	#     	X= request.files.get('file')
	# else:
	dataset=pd.read_csv(r"C:\Users\u22v03\Documents\Python Scripts\heart\heart.txt",header=None,sep=' ')
	dataset.columns=['Age', 'Gender', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG','Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal', 'Goal']
	# dataset=pd.read_csv(X,header=None,sep=' ')
	dataset=data_category(dataset)
	return dataset

def clean_data(dataset):
	dataset = pd.get_dummies(dataset,drop_first=False)
	#dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset)).values
	return dataset

def create_barplot(dataset,col):
	data = [go.Bar(x=dataset[col],y=list(range(20,100)),xbins=(30,70,1))]
	layout = go.Layout(autosize=False,width=850,height=400,title=go.layout.Title(text='Bar Plot',xref='paper',x=0),
	xaxis=dict(title=col,titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON
def create_scatterplot(dataset,col):
	data1 = [go.Scatter(x=dataset[col],y=dataset['Goal'],mode = 'markers',marker=dict(color = 'green'))]
	layout = go.Layout(autosize=False,width=850,height=400,title=go.layout.Title(text='Scatter Plot',xref='paper',x=0),
	xaxis=dict(title=col,titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'))
	graphJSON = json.dumps({'data':data1,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON
def create_countplot(dataset,col):
	data = [go.Histogram(x=dataset[col])]
	layout = go.Layout(autosize=False,width=1000,height=450,title=go.layout.Title(text='Histogram',xref='paper',x=0),
	xaxis=dict(title=col,titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'),
    yaxis=dict(title='Frequency',titlefont=dict(family='Arial, sans-serif',size=12,color='black'),showticklabels=True,tickangle=45,tickfont=dict(family='Old Standard TT, serif',size=10,color='black'),exponentformat='e',showexponent='all'))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def create_piechart(dataset,col):
	labels = dataset[col].value_counts().index
	values =dataset[col].value_counts().values
	data = [go.Pie(labels=labels, values=values)]
	layout = go.Layout(autosize=False,width=1000,height=450,title=go.layout.Title(text='Pie Chart',xref='paper',x=0))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON
def create_boxplot(dataset):
	data = []
	for col in dataset.columns:
		data.append(go.Box(y=dataset[col],name=col,showlegend=False,boxmean='sd'))
	layout = go.Layout(title=go.layout.Title(text='Box Plot',xref='paper',x=0),
	xaxis=dict(title='Columns'))
	graphJSON = json.dumps({'data':data,'layout':layout}, cls=plotly.utils.PlotlyJSONEncoder)
	return graphJSON

def lrmodel_parameter(X_train,X_test,y_train,y_test):
	clf = joblib.load('model1.pkl')
	probabilities1 = clf.predict_proba(X_test)
	output1 = clf.predict(X_test)
	cm2 = confusion_matrix(y_test, output1)
	train_accuracy=(clf.score(X_train, y_train))*100
	test_accuracy=(clf.score(X_test, y_test))*100
	sensitivity2 = (cm2[1,1]/(cm2[1,1]+cm2[1,0]))*100
	specificity2 = (cm2[0,0]/(cm2[0,0]+cm2[0,1]))*100
	precision2 = (cm2[1,1]/(cm2[1,1]+cm2[0,1]))*100
	F1score2=((2*sensitivity2*precision2)/(sensitivity2+precision2))
	auc_score=(roc_auc_score(y_test, output1))*100
	df={'Metrics':['Training_accuracy','Testing_accuracy','Sensitivity','Specificity','Precision','F1Score','AUC_Score'],'Score':[train_accuracy,test_accuracy,sensitivity2,specificity2,precision2,F1score2,auc_score]}
	return df

def rfmodel_parameter(X_train,X_test,y_train,y_test):
	rf = joblib.load('model2.pkl')
	probabilities2 = rf.predict_proba(X_test)
	output2 = rf.predict(X_test)
	cm2 = confusion_matrix(y_test, output2)
	train_accuracy=(rf.score(X_train, y_train))*100
	test_accuracy=(rf.score(X_test, y_test))*100
	sensitivity2 = (cm2[1,1]/(cm2[1,1]+cm2[1,0]))*100
	specificity2 = (cm2[0,0]/(cm2[0,0]+cm2[0,1]))*100
	precision2 = (cm2[1,1]/(cm2[1,1]+cm2[0,1]))*100
	F1score2=((2*sensitivity2*precision2)/(sensitivity2+precision2))
	auc_score=(roc_auc_score(y_test, output2))*100
	df={'Metrics':['Training_accuracy','Testing_accuracy','Sensitivity','Specificity','Precision','F1Score','AUC_Score'],'Score':[train_accuracy,test_accuracy,sensitivity2,specificity2,precision2,F1score2,auc_score]}
	return df

def dtmodel_parameter(X_train,X_test,y_train,y_test):
	dt = joblib.load('model.pkl')
	probabilities2 = dt.predict_proba(X_test)
	output2 = dt.predict(X_test)
	cm2 = confusion_matrix(y_test, output2)
	train_accuracy=(dt.score(X_train, y_train))*100
	test_accuracy=(dt.score(X_test, y_test))*100
	sensitivity2 = (cm2[1,1]/(cm2[1,1]+cm2[1,0]))*100
	specificity2 = (cm2[0,0]/(cm2[0,0]+cm2[0,1]))*100
	precision2 = (cm2[1,1]/(cm2[1,1]+cm2[0,1]))*100
	F1score2=((2*sensitivity2*precision2)/(sensitivity2+precision2))
	auc_score=(roc_auc_score(y_test, output2))*100
	df={'Metrics':['Training_accuracy','Testing_accuracy','Sensitivity','Specificity','Precision','F1Score','AUC_Score'],'Score':[train_accuracy,test_accuracy,sensitivity2,specificity2,precision2,F1score2,auc_score]}
	return df






@app.route("/")
@app.route("/home")
def home():
	dataset=load_data()
	df=dataset.head(10)
	return render_template('home.html',data=df.to_html())


@app.route('/dictionary')
def dictionary():
	return render_template("dictionary.html")

@app.route('/cleaning')
def cleaning():
	dataset=load_data()
	dataset1=dataset.drop(['Gender','Slope','Thal','CP','FBS','RestECG','Exang','Goal','CA'],axis=1)
	df=dataset1[~(np.abs(stats.zscore(dataset1)) < 3).all(axis=1)]
	plot1=create_boxplot(dataset1)
	df2 = pd.get_dummies(dataset,drop_first=True)
	df2 = (df2 - np.min(df2)) / (np.max(df2) - np.min(df2)).values
	df2=df2.head(5)
	return render_template("cleaning.html",plot1=plot1,data=df.to_html(),data2=df2.to_html())

@app.route('/visualisation')
def visualisation():
	path='static/graphs/correlation.png'
	return render_template("visualisation.html",path=path)

@app.route('/univariate_analysis')
def univariategraphs():
	dataset=load_data()
	dataset1=clean_data(dataset)
	col='Age'
	heading="Inferences"
	text="From the histogram, it is visible that most of the patient's age lies between 50 years to 65 years. The most patients are of age 59 years.Mean & Median of age distribution are 54.22 & 54.0	 Normal Test for the Age distribution NormaltestResult(statistic=6.747044110448095, pvalue=0.03426872819095869). From the above graphs and the skewness value and normal test, its clear that the distribution is not normal."
	bar1=create_countplot(dataset,col)
	return render_template("univariate_analysis.html",plot1=bar1,heading=heading,text=text)

@app.route('/change_column',methods=['POST'])
def change_graph():
	dataset=load_data()
	dataset1=clean_data(dataset)
	col=request.form.get('column')

	if col=='Age':
		bar1=create_countplot(dataset,col)
		heading="Inferences"
		text="From the histogram, it is visible that  the patient's age lies between 28 years to 77 years. Most of the patients are of age 59 years.Mean & Median of age distribution are 54.22 & 54.0.             	 Normal Test for the Age distribution NormaltestResult(statistic=6.747044110448095, pvalue=0.03426872819095869). From the above graphs and normal test, its clear that the distribution is not normal."
	if col=='Gender':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text='From the pie chart, it is clear that gender class is unbalanced . 70% of the patients are male and 30% are female. It is possible that male have more chance of having heart disease. '
	if col=='CP':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text='From the pie chart, it is clear that Chest Pain class is unbalanced . 50% of the patients have Asymptomatic Pain, 30% of patients have Non-anginal Pain, 15% of patients have Atypical Angina pain and rest of the patients have Typical Angina Pain.'
	if col=='Trestbps':
		bar1=create_countplot(dataset,col)
		heading="Inferences"
		text="From the histogram, it is visible that patient's resting blood pressure value lies between 90 to 204. Most of the patients have resting blood pressure value in range of  120 to 140 . Normal Test for the whole dataset.  NormaltestResult(statistic=12.148718536526424, pvalue=0.002301120114565709). This distribution is also not normal.As seen in the graph, the distribution contains two groups in it."
	if col=='Chol':
		bar1=create_countplot(dataset,col)
		text="From the histogram, it is visible that patient's cholestrol value lies between 120 to 420. Most of the patients have cholestrol value in range of  160 to 300 . There is an outlier with value of 560. Normal Test for the whole dataset NormaltestResult(statistic=2.10532832999989, pvalue=0.3490066979954109). This distribution is also not normal."
		heading="Inferences"
		
	if col=='FBS':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text='From the pie chart, it is clear that fasting blood sugar class is unbalanced . 85% of the patients have fasting blood sugar level greater than 120 mg/dl and 15% of patients have lesser than 120mg/dl. It is possible that patients with fasting blood sugar value greater than 120mg/dl have more chance of having heart disease.'
	if col=='RestECG':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text="From the pie chart, it is clear that Resting electrocardiographic results classes are unbalanced . 50% of the patients are Showing probable or definite left ventricular hypertropy by Estes criteria and 48% of the patients have normal graph and rest of the patients are having ST-T wave abnormality."
	if col=='Thalach':
		bar1=create_countplot(dataset,col)
		heading="Inferences"
		text="From the histogram, it is visible that patient's maximum heart rate value lies between 70 to 204. Most of the patients have maximum heart rate value in range of  140 to 170 . There is an outlier with value of 70.  Normal Test for the whole dataset NormaltestResult(statistic=2.10532832999989, pvalue=0.3490066979954109). This distribution is also not normal."
	if col=='Exang':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text="From the pie chart, it is clear that exercise induced angina classes are unbalanced . 67% of the patients didn't feel pain after exercise and rest of the patients felt pain after exercise."
	if col=='Oldpeak':
		bar1=create_countplot(dataset,col)
		heading="Inferences"
		text="From the histogram, it is visible that patient's oldpeak value lies between 0 to 4. Most of the patients have maximum heart rate value in range of  0 to 2 . There is an outlier with value of 6.Normal Test for the whole dataset NormaltestResult(statistic=2.10532832999989, pvalue=0.3490066979954109). This distribution is also not normal."
	if col=='Slope':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text="From the pie chart, it is clear that slope of the peak exercise ST segment classes are unbalanced . 48% of the patients have upsloping slope and 45% of the patients have flat slope and rest of the patients are having down-sloping."
	if col=='CA':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text="From the pie chart, it is clear that Resting number of major vessels classes are unbalanced . 59% of the patients have 0 major vessels and 21% of the patients have 1 major vessels and 12% of the patients have 2 major vessels and rest of the patients have 3 major vessels. "
	if col=='Thal':
		bar1=create_piechart(dataset,col)
		heading="Inferences"
		text="From the pie chart, it is clear that Thalassemia classes are unbalanced . 56% of the patients are showing normal defect and 38% of the patients have reversible defect and rest of the patients are having fixed defect."


	return render_template("univariate_analysis.html",plot1=bar1,heading=heading,text=text)

@app.route('/bivariate_analysis')
def bivariategraphs():
	path='static/graphs/'
	image1 = os.path.join(path, 'age.png')
	heading="Kdeplot : Age vs Goal"
	heading2="Central Limit Theorem"
	heading3="T-Test"
	texttest= "Assumption :- As seen in above graph, the mean of non-disease cohort is less than the mean of diseased cohort.<br>Null Hypothesis :- there is no difference in Mean of disease cohort and non-disease cohort.\nAlternate Hypothesis :- there is difference in Mean of disease cohort and non-disease cohort."
	texttest=texttest+"""t = 204.53902816158978  p = 0.0   p-value < 0.05 Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.So, I conclude that people who are slightly older have more chance of having heart disease. Therefore, age would be a predictive feature."""
		
	text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
	return render_template("bivariate_analysis.html", image1=image1,heading=heading,text=text,heading2=heading2,text2=texttest,heading3=heading3)


@app.route('/change_bicolumn',methods=['POST'])
def change_bigraph():
	col=request.form.get('column')
	path='static/graphs/'
	texttest= "Assumption :- As seen in above graph, the mean of non-disease cohort is less than the mean of diseased cohort.\\n Null Hypothesis :- there is no difference in Mean of disease cohort and non-disease cohort.\nAlternate Hypothesis :- there is difference in Mean of disease cohort and non-disease cohort."
	if col=='Age':
		image1 = os.path.join(path, 'age.png')
		heading="Kdeplot : Age vs Goal"
		text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
		heading2="Central Limit Theorem"
		text2=texttest+"""t = 204.53902816158978  p = 0.0   p-value < 0.05 Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.So, I conclude that people who are slightly older have more chance of having heart disease. Therefore, age would be a predictive feature."""
		heading3="T-Test"
	if col=='Gender':
		image1 = os.path.join(path, 'gender.png')
		heading="Gender vs Goal"
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition"
		text2="Chi statistics is 25.141967382205216 and  p value is 5.326104070129059e-07  As expected, given the low p-value(1.926225633356082e-06), so we reject null hypothesis and the test result detect a significant relationship between Gender and Goal. "
		heading3="Chi-square Test "
	if col=='CP':
		image1 = os.path.join(path, 'cp.png')
		heading="Chest Pain Type vs Goal"
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition "
		text2="Chi statistics is 63.94087517747943 and  p value is 8.450535705798324e-14 As expected, given the low p-value, so we reject null hypothesis and the test result detect a significant relationship between CP and Goal.Presence of disease graph have 90 patients with Chest pain type 4 much higher as compared to other chest pain.So, Asymptomatic pain can high predictive power. "
		heading3=" Chi-square Test "
	if col=='Trestbps':
		image1 = os.path.join(path, 'Trestbps.png')
		heading="Kdeplot : Resting Blood Pressure vs Goal"
		text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
		heading2="Central Limit Theorem"
		text2=texttest+"""t = 122.48573178423875   p = 0.0  p-value < 0.05  Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.The aggregated resting blood pressure for the entire dataset exhibited a mean value of 130 and for the diseased and non-diseased groups (i.e. 134 and 129 respectively).So, I conclude that people who have slightly high blood pressure have more chance of having heart disease.Therefore, resting blood pressure is a good predictive feature.  """
		heading3="T-Test"
	if col=='Chol':
		heading="Kdeplot : Cholestrol vs Goal"
		image1 = os.path.join(path, 'chol.png')
		text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
		heading2="Central Limit Theorem"
		text2=texttest+"""t = 146.98555862933068
p = 0.0   p-value < 0.05
Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.
Cholestrol levels for the non-disease cohort (median = 236 mg/dL) were lower compared to the diseased patients (median = 255 mg/dL) . Therefore, Cholestrol can be a good predictive feature."""
		heading3="T-Test"
	if col=='FBS':
		heading="Fasting Blood Sugar vs Goal"
		image1 = os.path.join(path, 'fbs.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test "
		text2="""Chi statistics is 0.00028776098726399107 and  p value is 0.986465718957874   As expected, given the high p-value, so we fail to reject null hypothesis and the test result detect a non-significant relationship between Fbs and Goal.
Most individuals did not have fasting blood sugar levels greater than 120 mg/dL. This did not change greatly when the data was divided based on the presence of disease.So, FBS is not a predictive feature."""
		heading3="Chi-square test Definition"
	if col=='Restecg':
		heading="Resting Electrocardiographic Results vs Goal"
		image1 = os.path.join(path, 'restecg.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition"
		text2="Chi statistics is 8.77219199941468 and  p value is 0.012449236295157253  Most patients exhibited normal resting electrocardiograhic results . However, a higher proportion of diseased patients had hypertropy suggesting that this feature may contribute some predictive power. "
		heading3="Chi-square test  "
	if col=='Thalach':
		heading="Kdeplot : Maximum Heart Rate vs Goal"
		image1 = os.path.join(path, 'thalach.png')
		text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
		heading2="Central Limit Theorem"
		text2=texttest+"""  t = -448.5440756371803
p = 0.0    p-value < 0.05
Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.
Maximum heart rate was higher for the disease cohort (mean =139 , median = 141) compared to non-disease patients (mean= 158 , median = 161). It was anticipated that this feature should have high predictive power."""
		heading3="T-Test"
	if col=='Exang':
		heading="Exercise Induced Angina vs Goal"
		image1 = os.path.join(path, 'exang.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition"
		text2=" Chi statistics is 45.094917186870525 and  p value is 1.8771197577079585e-11  As expected, given the low p-value, so we reject null hypothesis and the test result detect a significant relationship between Exang and Goal.Significantly more patients in the diseased cohort displayed exercise induced angina. This feature should be strongly predictive."
		heading3="Chi-square Test "
	if col=='Oldpeak':
		heading="Oldpeak vs Goal"
		image1 = os.path.join(path, 'oldpeak.png')
		text='To reduce the variablity around the means of two groups, I will use central limit theorem which states that given a sufficiently large sample size from a population with a finite level of variance, the mean of all sample from the same population will be approximately equal to the mean of the population.'
		heading2="Central Limit Theorem"
		text2=texttest+""" t = 458.3893613261764
p = 0.0   p-value < 0.05
Reject null hypothesis that there is no difference in Mean of disease cohort and non-disease cohort.
The Exercise Induced ST Depression differed between the non-disease and disease cohorts with the majority of cardiac disease patients exhibiting a higher mean and median for disease cohorts .Therefore, ST depression induced by exercise relative to rest can be a good predictive feature."""
		heading3="T-Test"
	if col=='Slope':
		heading="Slope vs Goal"
		image1 = os.path.join(path, 'slope.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition "
		text2="Chi statistics is 38.7797612721771 and  p value is 3.793864667948786e-09  As expected, given the low p-value, so we reject null hypothesis and the test result detect a significant relationship between Slope and Goal.Significantly more patients in the non-diseased cohort displayed Slope-Flat. This feature could be strongly predictive.The slope of the peak exercise ST segment differed between the non-disease and diseased cohorts with the majority of cardiac disease patients exhibiting a flat ST slope(value = 2).This can also have good predictive power. "
		heading3="Chi-square Test "
	if col=='CA':
		heading="Number of Major Vessels vs Goal"
		image1 = os.path.join(path, 'ca.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition"
		text2="Chi statistics is 60.40857401830753 and  p value is 4.807872016239094e-13  As expected, given the low p-value, so we reject null hypothesis and the test result detect a significant relationship between CA and Goal. Significantly more patients in the diseased cohort has number of blood vessels greater than 1. This feature should be strongly predictive. "
		heading3="Chi-square Test "
	if col=='Thal':
		heading="Thalassemia vs Goal"
		image1 = os.path.join(path, 'thal.png')
		text='The Chi Square statistic is commonly used for testing relationships between categorical variables.  The null hypothesis of the Chi-Square test is that no relationship exists on the categorical variables in the population; they are independent.'
		heading2="Chi-square Test Definition "
		text2="Chi statistics is 70.92692989782304 and  p value is 3.9665493643250104e-16  As expected, given the low p-value, so we reject null hypothesis and the test result detect a significant relationship between Thal and Goal. Significantly more patients in the diseased cohort has Reversible defect. This feature should be strongly predictive."
		heading3="Chi-square Test " 
	return render_template("bivariate_analysis.html", image1=image1,heading=heading,text=text,heading2=heading2,text2=text2,heading3=heading3)



@app.route('/logistic_prediction')
def logistic_prediction():
	dataset=load_data()
	dataset1=clean_data(dataset)
	X_train, X_test, y_train, y_test = train_test_split(dataset1.drop('Goal', 1), dataset1['Goal'], test_size = .2, random_state=42,shuffle=True)
	df_lr=lrmodel_parameter(X_train,X_test,y_train,y_test)
	df_lr=pd.DataFrame.from_dict(df_lr)
	df2={'HyperParameters':['Penalty','Class Weight','Inverse of Regularization Strength'],'Value':['Ridge Regularization','Class 0 : 1 <br /> Class 1 : 1', '0.3']}
	df2=pd.DataFrame.from_dict(df2)
	return render_template("logistic_prediction.html",data=df_lr.to_html(),data2=df2.to_html())

@app.route('/decision_prediction')
def decision_prediction():
	dataset=load_data()
	dataset1=clean_data(dataset)
	X_train, X_test, y_train, y_test = train_test_split(dataset1.drop('Goal', 1), dataset1['Goal'], test_size = .2, random_state=42,shuffle=True)
	df_dt=dtmodel_parameter(X_train,X_test,y_train,y_test)
	df_dt=pd.DataFrame.from_dict(df_dt)
	df2={'HyperParameters':['Criterion','Max Depth','Max Leaf Nodes','Min Samples Leaf','Min Samples Split'],'Value':['Entropy','3', '15','5','5']}
	df2=pd.DataFrame.from_dict(df2)
	return render_template("decision_prediction.html",data3=df_dt.to_html(),data2=df2.to_html())

@app.route('/random_prediction')
def random_prediction():
	dataset=load_data()
	dataset1=clean_data(dataset)
	X_train, X_test, y_train, y_test = train_test_split(dataset1.drop('Goal', 1), dataset1['Goal'], test_size = .2, random_state=42,shuffle=True)
	df_rf=rfmodel_parameter(X_train,X_test,y_train,y_test)
	df_rf=pd.DataFrame.from_dict(df_rf)
	df2={'HyperParameters':['Criterion','Max Depth','Max Features','Estimators'],'Value':['Gini','4', 'log2','500']}
	df2=pd.DataFrame.from_dict(df2)
	return render_template("random_prediction.html",data2=df_rf.to_html(),data3=df2.to_html())

@app.route('/form_prediction')
def form_prediction():
	return render_template("form_prediction.html")
@app.route('/form_prediction2',methods=['POST'])
def form_prediction2():
	Age=(request.form.get('Age'))
	RestECG=(request.form.get('RestECG'))
	CP=(request.form.get('CP'))
	Gender=(request.form.get('Gender'))
	Chol=(request.form.get('Chol'))
	Thalach=(request.form.get('Thalach'))
	FBS=(request.form.get('FBS'))
	Exang=(request.form.get('Exang'))
	Trestbps=(request.form.get('Trestbps'))
	Thal=(request.form.get('Thal'))
	Slope=(request.form.get('Slope'))
	CA=(request.form.get('CA'))
	Oldpeak=(request.form.get('Oldpeak'))
	list1=[Age,Gender,CP,Trestbps,Chol,FBS,RestECG,Thalach,Exang,Oldpeak,Slope,CA,Thal]
	list1=list(map(float, list1))
	list1=[list1]
	col = joblib.load('model_columns.pkl')
	old_col=['Age', 'Gender', 'CP', 'Trestbps', 'Chol', 'FBS', 'RestECG','Thalach', 'Exang', 'Oldpeak', 'Slope', 'CA', 'Thal']
	a = np.zeros(shape=(1,25))
	df2=pd.DataFrame(a,columns=col)
	df1=pd.DataFrame(list1,columns=old_col)
	df1=data_category2(df1)
	df1=clean_data(df1)
	for i in df1.columns:
		df2[i]=df1[i]


	clf = joblib.load('model1.pkl')
	probabilities1 = clf.predict_proba(df2)
	output1 = int(clf.predict(df2))

	dt = joblib.load('model.pkl')
	probabilities2 = dt.predict_proba(df2)
	output2 = int(dt.predict(df2))

	rf = joblib.load('model2.pkl')
	probabilities3 = rf.predict_proba(df2)
	output3 = int(rf.predict(df2))
	
	d={0:'Absence',1:'Presence'}
	mylist = [d[k] for k in [output1,output2,output3]]

	return render_template("form_prediction.html",output1=mylist[0],output2=mylist[1],output3=mylist[2])

if __name__ == '__main__':
	app.jinja_env.auto_reload=True
	app.config['TEMPLATES_AUTO_RELOAD']=True
	app.run(debug=True)
    
