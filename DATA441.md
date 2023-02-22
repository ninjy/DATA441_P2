## DATA 441 - Project 2
### By: Ninjin Gankhuleg

 1. Adapt and modify the code for Gramfort’s version of Lowess to accommodate train and test sets with multidimensional features.
 2. Test your new function from 1) on some real data sets with k-Fold cross-validations.
3. (Bonus 2 points) Create a SciKitLearn-compliant version of the function you wrote for 1) and test it with GridSearchCV from SciKitLearn.

### Libraries

    # graphical libraries
    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 120
    from IPython.display import Image
    from IPython.display import display
    plt.style.use('seaborn-white')
    !pip install --upgrade --q scipy
   
    # computational libraries
	`import numpy as np
	import pandas as pd
	from sklearn.linear_model import LinearRegression, Ridge
	from sklearn.preprocessing import StandardScaler, QuantileTransformer
	from sklearn.decomposition import PCA
	from scipy.spatial import Delaunay
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.pipeline import Pipeline
	import scipy.stats as stats
	from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
	from sklearn.metrics import mean_squared_error as mse
	from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
	from math import ceil
	from scipy import linalg
	from sklearn.base import BaseEstimator, RegressorMixin
	from sklearn.utils.validation import check_X_y, check_array, check_is_fitted`

## Cars Dataset
#### Scaling + Importing Data

    lm = LinearRegression()
	scale = StandardScaler()
	qscale = QuantileTransformer()
	
	data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Spring 23/DATA 441/data/cars.csv')

	x = data.loc[:,'CYL':'WGT'].values
	y = data['MPG'].values

	scale = StandardScaler()
	x = scale.fit_transform(x)

#### Function that computes the Euclidean distance between all the observations in u, and v

    def dist(u,v):
		if len(v.shape)==1:
			v = v.reshape(1,-1)
		d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in  range(len(v))])
		return d
### Gramfort’s version of Lowess to accommodate train and test sets with multidimensional features
    def L_AG_MD(x, y, xnew,f=2/3,iter=3, intercept=True):
    
		n = len(x)
		r = int(ceil(f * n))
		yest = np.zeros(n)
		
		if  len(y.shape)==1: # here we make column vectors
			y = y.reshape(-1,1)
			
		if  len(x.shape)==1:
			x = x.reshape(-1,1)
			
		if intercept:
			x1 = np.column_stack([np.ones((len(x),1)),x])
		else:
			x1 = x
		
		h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in  range(n)]
		
		w = np.clip(dist(x,x) / h, 0.0, 1.0)
		w = (1 - w ** 3) ** 3
		
		#Looping through all X-points
		delta = np.ones(n)
		for iteration in  range(iter):
			for i in  range(n):
				W = np.diag(delta).dot(np.diag(w[:,i]))
				b = np.transpose(x1).dot(W).dot(y)
				A = np.transpose(x1).dot(W).dot(x1)
			
				A = A + 0.0001*np.eye(x1.shape[1]) 
				beta = linalg.solve(A, b)
				yest[i] = np.dot(x1[i],beta)
				
			residuals = y.ravel() - yest
			s = np.median(np.abs(residuals))
			delta = np.clip(residuals / (6.0 * s), -1, 1)
			delta = (1 - delta ** 2) ** 2
			
		if x.shape[1]==1:
			f = interp1d(x.flatten(),yest,fill_value='extrapolate')
			output = f(xnew)
		else:
			output = np.zeros(len(xnew))
			for i in  range(len(xnew)):
				ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
				pca = PCA(n_components=2)
				x_pca = pca.fit_transform(x[ind])
				tri = Delaunay(x_pca,qhull_options='QJ')
				f = LinearNDInterpolator(tri,yest[ind])
				output[i] = f(pca.transform(xnew[i].reshape(1,-1)))
		if  sum(np.isnan(output))>0:
			g = NearestNDInterpolator(x,y.ravel())
			output[np.isnan(output)] = g(xnew[np.isnan(output)])
		return output


	xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
	xtrain = scale.fit_transform(xtrain)
	xtest = scale.transform(xtest)
	yhat = L_AG_MD(xtrain,ytrain,xtest,f=1/3,iter=3,intercept=True)

### Scikit-Learn Compliant Function Version
	class Lowess_AG_MD:
		def __init__(self, f = 1/10, iter = 3,intercept=True):
			self.f = f
			self.iter = iter
			self.intercept = intercept

		def fit(self, x, y):
			f = self.f
			iter = self.iter
			self.xtrain_ = x
			self.yhat_ = y
		  
		def predict(self, x_new):
			check_is_fitted(self)
			x = self.xtrain_
			y = self.yhat_
			f = self.f
			iter = self.iter
			intercept = self.intercept
			return L_AG_MD(x, y, x_new, f, iter, intercept)

		def get_params(self, deep=True):
			return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

		def set_params(self, **parameters):
			for parameter, value in parameters.items():
				setattr(self, parameter, value)
			return  self

### K-Fold Cross-Validations
	mse_lwr = []
	mse_rf = []
	kf = KFold(n_splits=10,shuffle=True,random_state=1234)
	model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
	model_lw = Lowess_AG_MD(f=1/3,iter=2,intercept=True)

	for idxtrain, idxtest in kf.split(x):
		xtrain = x[idxtrain]
		ytrain = y[idxtrain]
		ytest = y[idxtest]
		xtest = x[idxtest]
		xtrain = scale.fit_transform(xtrain)
		xtest = scale.transform(xtest)

		model_lw.fit(xtrain,ytrain)
		yhat_lw = model_lw.predict(xtest)

		model_rf.fit(xtrain,ytrain)
		yhat_rf = model_rf.predict(xtest)

		mse_lwr.append(mse(ytest,yhat_lw))
		mse_rf.append(mse(ytest,yhat_rf))

	print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
	print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))

The Cross-validated Mean Squared Error for Locally Weighted Regression is : 16.37831448828224 
The Cross-validated Mean Squared Error for Random Forest is : 17.152729883572956

Based on the output, the MSE for Locally Weighted Regression is smaller than the Random Forest, which indicates that Locally Weighted Regression is doing a better job of fitting the data.

### Grid Search CV
	lwr_pipe = Pipeline([('zscores', StandardScaler()),
	('lwr', Lowess_AG_MD())])

	params = [{'lwr__f': [1/i for i in  range(3,15)],
	'lwr__iter': [1,2,3,4]}]

	gs_lowess = GridSearchCV(lwr_pipe,
	param_grid=params,
	scoring='neg_mean_squared_error',
	cv=5)
	
	gs_lowess.fit(x, y)
	gs_lowess.best_params_

The best hyperparameters are outputted are:
{'lwr__f': 0.3333333333333333, 'lwr__iter': 2}

	gs_lowess.score(x,y)

The mean squarred error of the fitted pipeline on the input data `x` and `y` is:
-15.324988522298963


## Concrete Dataset
#### Scaling + Importing Data

    data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Spring 23/DATA 441/data/concrete.csv')

	x = data.loc[:,'cement':'age'].values
	y = data['strength'].values

	scale = StandardScaler()
	x = scale.fit_transform(x)

	xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
	xtrain = scale.fit_transform(xtrain)
	xtest = scale.transform(xtest)
	yhat = L_AG_MD(xtrain,ytrain,xtest,f=1/3,iter=3,intercept=True)

Since we already defined the dist and L_AG_MD functions, and the Lowess_AG_MD class before, we won't redefine it for the Concrete dataset.

### K-Fold Cross-Validations
	mse_lwr = []
	mse_rf = []
	kf = KFold(n_splits=10,shuffle=True,random_state=1234)
	model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)
	model_lw = Lowess_AG_MD(f=1/3,iter=2,intercept=True)

	for idxtrain, idxtest in kf.split(x):
		xtrain = x[idxtrain]
		ytrain = y[idxtrain]
		ytest = y[idxtest]
		xtest = x[idxtest]
		xtrain = scale.fit_transform(xtrain)
		xtest = scale.transform(xtest)

		model_lw.fit(xtrain,ytrain)
		yhat_lw = model_lw.predict(xtest)

		model_rf.fit(xtrain,ytrain)
		yhat_rf = model_rf.predict(xtest)

		mse_lwr.append(mse(ytest,yhat_lw))
		mse_rf.append(mse(ytest,yhat_rf))

	print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
	print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
	
The Cross-validated Mean Squared Error for Locally Weighted Regression is : 44.93819920178596 
The Cross-validated Mean Squared Error for Random Forest is : 45.157001913168806

Based on the output, the MSE for Locally Weighted Regression is smaller than the Random Forest, which indicates that Locally Weighted Regression is doing a better job of fitting the data.

### Grid Search CV
	lwr_pipe = Pipeline([('zscores', StandardScaler()),
	('lwr', Lowess_AG_MD())])

	params = [{'lwr__f': [1/i for i in  range(3,15)],
	'lwr__iter': [1,2,3,4]}]

	gs_lowess = GridSearchCV(lwr_pipe,
	param_grid=params,
	scoring='neg_mean_squared_error',
	cv=5)
	
	gs_lowess.fit(x, y)
	gs_lowess.best_params_

The best hyperparameters are outputted are:
{'lwr__f': 0.07142857142857142, 'lwr__iter': 1}


	gs_lowess.score(x,y)

The mean squarred error of the fitted pipeline on the input data `x` and `y` is:
-1.6198492233009538

