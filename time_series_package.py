import numpy as np 
import random
from scipy.fftpack import fft
import sklearn 
import matplotlib.pyplot as plt 
import warnings


### Global Variables
## prediction length 
PRED_LEN=60 
WARN_NUM=0

#########################################  Time series visualization  ########################################

## sample a time series from N time series 
def tssample(ts,id=False):
	N=ts.shape[0]
	row=random.randint(0,N-1)
	if 	id==False:
		return ts[row,:]
	else:
		return row 

## ts: a T vector or a N*T matrix with each row representing a time series 

## plot a single time series 
def tsplot(ts, test=None, pred=None, id=None, spikes=None, trend=None, plot_type='.', t_mark=True):
	fig=plt.figure(1,figsize=[16,5])
	if pred is None:
		if spikes is None:
			plt.plot(ts,plot_type,label='training data')
			if trend is not None:
				plt.plot(trend,label='trend',alpha=0.5)
		else: 
			plt.plot(ts,label='spikes',alpha=0.3)
			plt.plot(ts-spikes,label='no spikes signal')
			if trend is not None:
				plt.plot(trend,label='trend',alpha=0.5)
	else:
		n1=len(ts)
		n2=len(pred)
		x_max=n1+n2
		x1=np.arange(n1)
		x2=n1+np.arange(n2)
		x=np.arange(x_max)
		y_max=max(max(ts),max(pred))
		if test is not None:
			y_max=max(y_max,max(test))
		plt.figure(1,figsize=[16,5])
		plt.axis([0,x_max,0,y_max*1.05])
		if spikes is None:
			plt.plot(x1,ts,plot_type,label="traning data")
		else:
			plt.plot(x1,ts,label='spikes',alpha=0.3)
			plt.plot(x1,ts-spikes,label='spikeless signal')
		plt.plot(x2,pred,label='prediction',linewidth=3)
		if test is not None:
			plt.plot(x2,test,label='test data',alpha=0.7)
		if trend is not None:
			plt.plot(x1,trend,label='trend',alpha=0.5)
		if id is not None:
			plt.title("Time series: %d" %id)
		else:
			plt.title("Time series")
		if t_mark:
			plt.axvline(n1-365, alpha=0.5,color='red')
			plt.axvline(n1,alpha=0.5, color='red')
			plt.axvline(n1-91, alpha=0.5, color='red')
			plt.axvline(n1-183,alpha=0.5, color='red')
			plt.axvline(n1-273, alpha=0.5, color='red')
	plt.show()


### Preprocessing 
## data split 
def get_training_data(ts,test_len=PRED_LEN):
	return ts[:,:-PRED_LEN]

def get_test_data(ts,test_len=PRED_LEN):
	return ts[:,-PRED_LEN:] 

## take the last no-NaN part of time series 
def remove_na(ts):
	if ts.ndim==1:
		na_idx=np.isnan(ts).nonzero()[0]
		if len(na_idx)==0:
			return ts 
		start_idx=max(na_idx)+1
		return ts[start_idx:]
	else:
		print("Haven't implemented multi-time series case! :( ")
		raise NotImplementedError

## count zeros from of ts[begin:end]
def count_zeros(ts, win_len='full'):
	if ts.ndim==1:
		T=len(ts)
		if win_len=='full' or win_len>T:
			win_len=T 
		return np.sum(ts[-win_len]==0)
	else:
		T=ts.shape[1]
		if win_len=='full' or win_len>T:
			win_len=T 
		return np.sum(ts[:,-win_len:]==0,axis=1)

 
############################################# Smooth and Trend #####################################################

def smooth(ts,win_len=49,method='median'):
	if method== 'median':
		return med_smooth(ts,win_len)
	elif method== 'mean':
		return ave_smooth(ts,win_len)
	else:
		raise NotImplementedError

def ave_smooth(ts,win_len=49):
	half_win=int(win_len/2)
	if ts.ndim==1:
		T=len(ts)
		trend=np.zeros(T)
		for i in range(T):
			if i<half_win:
				trend[i]=np.mean(ts[:win_len])
			elif i<T-half_win:
				trend[i]=np.mean(ts[i-half_win:i+half_win+1])
			else:
				trend[i]=np.mean(ts[i+1-win_len:i+1])
	elif ts.ndim==2:
		T=ts.shape[1]	
		print("Indentifying trend ...") 
		trend = np.zeros(ts.shape)
		for i in range(T):
			if i<half_win:
				trend[:,i]=np.mean(ts[:,:win_len],axis=1)
			elif i<T-half_win:
				trend[:,i]=np.mean(ts[:,i-half_win:i+half_win+1],axis=1)
			else:
				trend[:,i]=np.mean(ts[:,i+1-win_len:i+1],axis=1)
	return trend;


def med_smooth(ts,win_len=49):
	half_win=int(win_len/2)
	if ts.ndim==1:
		T=len(ts)
		trend=np.zeros(T)
		for i in range(T):
			if i<half_win:
				trend[i]=np.median(ts[:win_len])
			elif i<T-half_win:
				trend[i]=np.median(ts[i-half_win:i+half_win+1])
			else:
				trend[i]=np.median(ts[i+1-win_len:i+1])
	elif ts.ndim==2:
		T=ts.shape[1]	
		print("Indentifying trend ...") 
		trend = np.zeros(ts.shape)
		for i in range(T):
			if i<half_win:
				trend[:,i]=np.median(ts[:,:win_len],axis=1)
			elif i<T-half_win:
				trend[:,i]=np.median(ts[:,i-half_win:i+half_win+1],axis=1)
			else:
				trend[:,i]=np.median(ts[:,i+1-win_len:i+1],axis=1)
	return trend;

def get_trend(trend, pred_len=60, method='-'):
	if trend.ndim==1:
		if method=='-':
			return np.ones(pred_len)*trend[-1]
		else:
			raise NotImplementedError
	else:
		N=trend.shape[0]
		if method is '-':   # take the last value in "trend" as estimation 
			pred_trend = np.tile(trend[:,-1],(pred_len,1)).T
			return pred_trend 
		else:
			raise NotImplementedError  

def decay_est(ts, new_data_weight=0.2, win_len=49, pred_len=PRED_LEN):
	if ts.ndim==2:
		N=ts.shape[0]
		T=ts.shape[1]
		est=np.zeros(N)
		for i in range(win_len-1,0,-1):  # i=win_len-1, win_len-2, ..., 1 
			est=ts[:,-i]*new_data_weight+est*(1-new_data_weight)
		return np.tile(est,(pred_len,1)).T
	else:
		raise NotADirectoryError 


def trend_check(ts, threshold=0.8, win_len=50, log=True , coef='all'): 
	if ts.ndim==1:
		ts=[ts]
	N=ts.shape[0]
	y=ts[:,-win_len:]
	if log:
		y=np.log(y+1)
	x=np.tile(np.arange(win_len),(N,1))
	n=win_len
	# Model: y(t)=kx+b 
	p=np.sum(x*y,axis=1)-n*np.mean(x,axis=1)*np.mean(y,axis=1)
	q=np.sum(x*x,axis=1)-n*np.mean(x,axis=1)*np.mean(x,axis=1)
	p=np.array(p,dtype='float64')
	k=p/q 
	b=np.mean(y,axis=1)-k*np.mean(x,axis=1)
	residual=y-np.einsum('i,ij->ij',k,x)-np.tile(b,(n,1)).T
	total_var=np.var(y,axis=1)
	exp_var=total_var-np.var(residual,axis=1)
	total_var[total_var==0]=1
	Rsq=exp_var/total_var 
	if coef=='all':
		return Rsq>threshold 
	elif coef=='positive':
		return np.bitwise_and(Rsq>threshold, k>0)
	elif coef=='negative':
		return np.bitwise_and(Rsq>threshold, k<0)
	else:
		raise RuntimeError("Argument coef can't be %s" %coef)

def trend_pred(ts, pred_len=PRED_LEN, win_len=50, log=True):
	if ts.ndim==1:
		ts=[ts]
	N=ts.shape[0]
	y=ts[:,-win_len:]
	if log:
		y=np.log(y+1)
	x=np.tile(np.arange(win_len),(N,1))
	n=win_len
	# Model: y(t)=kx+b 
	p=np.sum(x*y,axis=1)-n*np.mean(x,axis=1)*np.mean(y,axis=1)
	q=np.sum(x*x,axis=1)-n*np.mean(x,axis=1)*np.mean(x,axis=1)
	p=np.array(p,dtype='float64')
	k=p/q
	b=np.mean(y,axis=1)-k*np.mean(x,axis=1)
	x_pred=np.tile(np.arange(pred_len)+win_len,(N,1))
	y_pred=np.einsum('i,ij->ij',k,x_pred)+np.tile(b,(pred_len,1)).T 
	if log:
		y_pred=np.exp(y_pred)-1
	return y_pred 


#################################################### Spike Processing #####################################################
## find spikes position
## potential problems: only apply to time series with the same length (how to deal with "nan"s?)
def find_spikes(ts, threshold=7, method='same'):
	if ts.ndim==1:
		T=len(ts)
		if method=='same':
			sd=np.std(ts)   # sd is a scalar 
			spikes_pos=(ts>threshold*sd).nonzero()
			return spikes_pos
	elif ts.ndim==2:    
		N=ts.shape[0]
		T=ts.shape[1]
		if method=='same':
			sd=np.std(ts,axis=1)
			sd=np.tile(sd,[T,1]).T
			spikes_pos=(ts>threshold*sd).nonzero()
			return spikes_pos 

## get the spikes 
def get_spikes(ts,spikes_pos):
	spikes=np.zeros(ts.shape)
	spikes[spikes_pos]=ts[spikes_pos]
	return spikes

'''
## remove spikes with 0 or designated value filled in the spike positions 
def remove_spikes(ts,spikes_pos,replace_val=0):
	if ts.ndim==1:
		T=len(ts)
		if len(replace_val)==1:
			replace_val=np.ones(T)*replace_val
		ts[spikes_pos]=replace_val[spikes_pos]
		return ts 
	else:
		N=ts.shape[0]
		T=ts.shape[1]
		if replace_val is int 
'''

################################################## Periodicity ###########################################################
## check yearly periodicity 
def year_check(ts, threshold=0.8, intercept=True): 
	if ts.ndim==1:
		ts=np.array([ts])
	x=ts[:,:-365]
	y=ts[:,365: ]
	if intercept==False: 	# model: y=kx 
		q=np.sum(x*x,axis=1) # denominator 
		p=np.sum(x*y,axis=1) # numerator 
		q[q==0]=1 
		p=np.array(p,dtype="float64")
		k=p/q
		residule=y-np.einsum('i,ij->ij',k,x)
	else:					# model: y=kx+b
		n=x.shape[1]
		q=np.sum(x*x,axis=1)-n*np.mean(x,axis=1)*np.mean(x,axis=1)
		p=np.sum(x*y,axis=1)-n*np.mean(x,axis=1)*np.mean(y,axis=1)
		q[q==0]=1
		p=np.array(p,dtype="float64")
		k=p/q
		b=y.mean(axis=1)-k*x.mean(axis=1)
		residule=y-np.einsum('i,ij->ij',k,x)-np.tile(b,(n,1)).T 
	total_var=np.var(y,axis=1)
	exp_var=total_var-np.var(residule,axis=1)  # explained variance 
	total_var[total_var==0]=1
	Rsq=exp_var/total_var
	return Rsq>threshold

## test year_check()  
ts=np.random.randn(10,730)
ts[:,:365]=ts[:,365:]*np.random.randn()+np.tile(np.random.randn(10),(365,1)).T
assert year_check(ts,threshold=0.95).all()==True



def year_pred(ts, pred_len=60, intercept=True):  
	takeout=False 
	if ts.ndim==1:
		ts=np.array([ts])
		takeout=True 

	N=ts.shape[0]
	x=ts[:,:-365]
	y=ts[:,365: ]

	x_pred=ts[:,-365:-365+pred_len]
	if intercept==False: 	# model: y=kx 
		q=np.sum(x*x,axis=1) # denominator 
		p=np.sum(x*y,axis=1) # numerator 
		q[q==0]=1 
		p=np.array(p,dtype="float64")
		k=p/q
		y_pred=np.einsum('i,ij->ij',k,x_pred)
	else:					# model: y=kx+b
		n=x.shape[1]
		q=np.sum(x*x,axis=1)-n*np.mean(x,axis=1)*np.mean(x,axis=1)
		p=np.sum(x*y,axis=1)-n*np.mean(x,axis=1)*np.mean(y,axis=1)
		q[q==0]=1
		p=np.array(p,dtype="float64")
		k=p/q
		b=y.mean(axis=1)-k*x.mean(axis=1)
		y_pred=np.einsum('i,ij->ij',k,x_pred)+np.tile(b,(pred_len,1)).T
	if takeout:
		return y_pred[0]
	else:
		return y_pred  


## weekly perodicity detection 
def week_check(ts, win_len="full", threshold=2):
	if ts.ndim==1:
		T=len(ts)

		# make win_len multiple of 7 
		if win_len == "full":
			win_len = (T//7)*7
		else:
			if win_len>T:
				win_len=T 
			win_len = (win_len//7)*7 
		# Fourier transform 
		f_complex=fft(ts[-win_len:])    
		f_mag=np.abs(f_complex)
		weekf=win_len/7  
		# choose smooth window 
		if(weekf>=50):
			s_win=10
		elif(weekf>=10):
			s_win=5
		else:
			s_win=weekf/2
		# smooth in frequency domain 
		weekf=int(weekf)
		f_begin=int(weekf-s_win)   # here we choose 11 to be smooth window 
		f_end=int(weekf+s_win)	
		ave_mag=np.mean(f_mag[f_begin:f_end])  # average magnitude 
		return f_mag[weekf]>threshold*ave_mag
	else:
		print("Identifying periodicity...")
		N=ts.shape[0]
		T=ts.shape[1]
		# make win_len multiple of 7 
		if win_len == "full":
			win_len = (T//7)*7
		else:
			if win_len>T:
				win_len=T 
			win_len = (win_len//7)*7 
		# Fourier transform
		f_complex=fft(ts[:,-win_len:],axis=1)
		f_mag=np.abs(f_complex)
		weekf=win_len/7
		# choose smooth window 
		if(weekf>=50):
			s_win=10
		elif(weekf>=10):
			s_win=10
		else:
			s_win=weekf//2
		# smooth in frequency domain 
		weekf=int(weekf)
		f_begin=int(weekf-s_win)
		f_end=int(weekf+s_win)
		ave_mag=np.mean(f_mag[:,f_begin:f_end],axis=1) # a N-vector 
		return f_mag[:,weekf]>threshold*ave_mag  # return a N-vector of (True or False)

# weekly periodicity prediction using Fourier transform 	
def week_pred(ts, pred_len=PRED_LEN, win_len=84, waveNum='all', week_exist=None):
	if ts.ndim==1:
		T=len(ts)
		# make win_len multiple of 7 
		if win_len == "full":
			win_len = (T//7)*7
		else:
			if win_len>T:
				win_len=T 
			win_len = (win_len//7)*7 
		ts=ts[-win_len:]

		# Fourier transform 
		f_complex=fft(ts)    
		f_mag=np.abs(f_complex)

		weekf=win_len/7  
		if waveNum=='all':
			waveNum=6

		week_coef=np.array(weekf * (1+np.arange(waveNum)), dtype='int')
		t=np.arange(pred_len)+win_len  
		pred=np.zeros(pred_len)
		for k in week_coef:
			phase=np.angle(f_complex[k])
			pred+=f_mag[k]*np.cos( 2* np.pi *t*k / win_len+phase )/win_len 
		return pred
	else:
		print("Predicting periodicity...")
		N=ts.shape[0]
		T=ts.shape[1]
		if win_len == "full":
			win_len = (T//7)*7
		else:
			if win_len>T:
				win_len=T 
			win_len = (win_len//7)*7 
		ts=ts[:,-win_len:]
		f_complex=fft(ts,axis=1)
		f_mag=np.abs(f_complex)

		weekf=win_len/7
		if waveNum=='all':
			waveNum=6
		week_coef=np.array(weekf*(1+np.arange(waveNum)),dtype='int')
		if week_exist is None:
			week_exist=np.arange(N)
		# initialization 
		pred=np.zeros([N,pred_len]) 
		t=np.arange(pred_len)+win_len
		t=np.tile(t,(sum(week_exist),1)) ## sum(week_exist)= number of "True" in week_exist 
		for k in week_coef:
			phase=np.angle(f_complex[week_exist,k])
			phase=np.tile(phase,(pred_len,1)).T 
			pred[week_exist,:]+=np.einsum('i,ij->ij',f_mag[week_exist,k],np.cos(2*np.pi*k*t/win_len+phase))/win_len
		return pred


## get the weekly periodicity as a periodic function from [-win_len: -0]
def get_week_pero(ts, win_len="full", waveNum='all', week_exist=None):
	if ts.ndim==2: 
		print("Predicting periodicity...")
		N=ts.shape[0]
		T=ts.shape[1]
		if win_len == "full":
			win_len = (T//7)*7
		else:
			if win_len>T:
				win_len=T 
			win_len = (win_len//7)*7 
		ts=ts[:,-win_len:]
		f_complex=fft(ts,axis=1)
		f_mag=np.abs(f_complex)

		weekf=win_len/7
		if waveNum=='all':
			waveNum=6
		week_coef=np.array(weekf*(1+np.arange(waveNum)),dtype='int')
		if week_exist is None:
			week_exist=np.arange(N)

		# initialization 
		pero=np.zeros([N,win_len]) 
		t=np.arange(win_len)
		t=np.tile(t,(sum(week_exist),1)) ## sum(week_exist)= number of "True" in week_exist 
		for k in week_coef:
			phase=np.angle(f_complex[week_exist,k])
			phase=np.tile(phase,(win_len,1)).T 
			pero[week_exist,:]+=np.einsum('i,ij->ij',f_mag[week_exist,k],np.cos(2*np.pi*k*t/win_len+phase))/win_len
		return pero 

## remove the weekly periodicity from [-win_len:-0]  
def remove_week_pero(ts,win_len="full",waveNum="all",week_exist=None):
	pero=get_week_pero(ts, win_len=win_len, waveNum=waveNum, week_exist=week_exist) 
	ts[:,-win_len:]-=pero 
	return ts 

###################################################### Error Processing ##########################################################
## error prediction 
def cat_smooth(ts,pred,rate=0.5):
	err=ts[:,-1]-pred[:,0]
	for i in range(pred.shape[1]):
		err=err*rate
		pred[:,i]=pred[:,i]+err
	return pred 

def error_pred(error,reg_len=6,win_len=98,pred_len=PRED_LEN, method="lasso",alpha=1.0):
	if error.ndim==2:
		N=error.shape[0]
		T=error.shape[1]
		pred_error=np.zeros([N,pred_len])
		if method== "simple":  # error decay with rate 0.5 
			err=error[:,-1]   # start 
			for i in range(pred_len):
				pred_error[:,i]=err*0.5
				err=err*0.5
			return pred_error

		if (win_len is "full") or (win_len>T):
			win_len=T 
		error=error[:,-win_len:]   # remove the irrelevant part 
		print("Fitting %s regression models ... " %method)
		for i in range(N):
			pred_error[i,:]=error_pred(error[i,:], reg_len=reg_len, win_len="full", pred_len=pred_len, method=method, alpha=alpha)
			if i%10000 == 0:
				print("Finish %d time series ..." %i)
		return pred_error
	else:
		T=len(error)
		if (win_len is "full") or (win_len>T):
			win_len=T 
		error=error[-win_len:]     # remove the irrelevant part 
		pred_error=np.zeros(pred_len)
		Y=error[reg_len:]
		X=[]
		for i in range(win_len-reg_len):	
			X.append(error[i:i+reg_len])
		X=np.array(X)
		# print(X)
		if method=="lasso":
			clf=sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=False,max_iter=10000)
		global WARN_NUM 
		with warnings.catch_warnings():
			warnings.filterwarnings('error') 
			try: 
				clf.fit(X,Y)
			except Warning: 
				WARN_NUM+=1
				print("Warning index: %d" %WARN_NUM)
				return pred_error 
		x=error[-reg_len:]
		for i in range(pred_len):
			pred_error[i]=clf.predict(x[np.newaxis,:])
			x=np.append(x[1:],pred_error[i])
		return pred_error 


############################################## SMAPE iteration ###################################################

## target 
'''
def lrdiff(x,a):
	f=x/((a+x)*(a+x))   # a vector 
	return np.sum(f*(np.array(x<a, dtype='int')-np.array(x>=a,dtype='int'))) 


def argmin_smape(ts):
	# target: find best a so that sum(smape(a,x[i])) is minimized 
	if ts.ndim==1:
		# initialize a:
		x=ts 
		if (x==x[0]).all():
			return x[0]
		# initialize a 
		a=np.round(np.median(x))
		while(lrdiff(x,a)<=0):
			a=a+1
		while(lrdiff(x,a)>0):
			a=a-1
		if np.abs(lrdiff(x,a))>np.abs(lrdiff(x,a+1)):
			return a+1
		else:
			return a
	else:
		N=ts.shape[0]
		a=np.zeros(N)
		for i in range(N):
			x=ts[i,:]
			a[i]=argmin_smape(x)
			if i%1000==0:
				print("Processing %d-th time series" %i)
		return a
'''

## @param: ts: a vector 
## @param: a : a scalar 
def mean_smape(ts,a):  
	assert (ts>=0).all()
	assert a>=0
	if a==0:
		return np.mean(ts!=0)*2 
	return np.mean(np.abs(ts-a)/(ts+a))*2



def greed_argmin(ts):   # find argmin_{a} smape(a,ts)
	if ts.ndim==1:
		a=np.round(np.median(ts)) # initialize 
		while mean_smape(ts,a)>mean_smape(ts,a+1):
			a+=1
		if a==0:
			return 0
		while mean_smape(ts,a)>mean_smape(ts,a-1):
			a-=1
			assert a>=0
			if a==0:
				return 0
		if mean_smape(ts,a)<mean_smape(ts,0):
			return a 
		else:
			return 0
	else:
		N=ts.shape[0]
		a=np.zeros(N)
		for i in range(N):
			x=ts[i,:]
			a[i]=greed_argmin(x)
			if i%1000==0:
				print("Processing %d-th time series" %i)
		return a 


############################################### Final Predict #####################################################

### Generate Prediction 
## @Param: pred_len=60, length of prediction 
## @Param: smooth_win_len --- window size when median is taken as trend 
## @Param: year_threshold --- R square threshold when identifying yearly trend 
## @Param: spike_threshold --- threshold when identifying spikes (measured by number of std)
## @Param: week_global_threshold --- threshold when identifying global week periodicity 
## @Param: week_local_threshold --- threshold when identifying local week periodicity 
## @Param: week_win_len --- window size when identifying and predicting local week periodicity 
## @Param: 
	

def predict(ts, pred_len=PRED_LEN, smooth_win_len=49, year_threshold=0.5, spike_threshold=7, week_global_threshold=5, week_local_threshold=3, 
	week_win_len=98):
	if ts.ndim==2:
		N=ts.shape[0]
		T=ts.shape[1]

		## count zero for last PRED_LEN(=60) days 
		zero_num= count_zeros(ts,win_len=PRED_LEN)

		## rough smooth 
		trend=smooth(ts, win_len= smooth_win_len, method='median')	
		pred_trend_1= get_trend(trend, method='-')    # raw median prediction

		## Remove spikes 
		detrend_sig = ts -trend  # detrend signal 
		spikes_pos = find_spikes(detrend_sig, threshold= spike_threshold, method='same')   # get spike positions 
		r_sig=detrend_sig 	  # detrend, despikes signal 
		r_sig[spikes_pos]=0   # replace spikes value with 0 (i.e. trend value in the original signal)
		ts_ns = trend +r_sig  # time series, no spikes 

		## trend re-evaluation with decaying influence  
		pred_trend_2 = decay_est(ts_ns, new_data_weight=0.2, win_len=49)

		'''
		## processing linear trend 
		lin_pos_decrease = trend_check(ts, win_len=50, threshold=0.6, log=True,  coef="negative")
		lin_pos_increase = trend_check(ts, win_len=50, threshold=0.6, log=False, coef="positive")
		pred_trend_4 = pred_trend_3.copy()	# initialization 
		pred_trend_4[lin_pos_decrease,:]= trend_pred(ts[lin_pos_decrease,:], pred_len=60, log=True)
		pred_trend_4[lin_pos_increase,:]= trend_pred(ts[lin_pos_increase,:], pred_len=60, log=False) 
		'''

		## trend aggregation 
		# pred_trend = pred_trend_1*0.66 + pred_trend_2*0.34
		pred_trend = np.tile(greed_argmin(ts[:,-49:]),(60,1)).T 	

		## Process periodicity 
		week_global = week_check(r_sig ,win_len='full', threshold= week_global_threshold)
		week_local  = week_check(r_sig ,win_len=week_win_len, threshold = week_local_threshold) 

		week_pos = week_global | week_local  # we think there is week trend either locally there is a weekly periodicity or globally there is a weekly periodicity	 
		# week_pos = week_local 
		pred_week = week_pred(r_sig, win_len=week_win_len, waveNum="all", week_exist=week_pos)  # predict weekly periodicity with local intensity 

		## Fit error 
		pred_error = np.zeros_like(pred_week)
		'''
		error=remove_week_pero(r_sig, win_len=week_win_len, waveNum="all", week_exist=week_pos)
		pred_error= error_pred(error,reg_len=10	,win_len=week_win_len, method="lasso", alpha=10)
		pred_error= error_pred(ts-trend, method="simple") 
		'''
		
		## processing yearly correlation with original data 
		year_pos=year_check(ts,threshold=year_threshold)  # positions that yearly trend is significant 	

		## Generate prediction 
		pred=pred_trend+pred_week+pred_error
		pred[year_pos,:]=year_pred( ts[year_pos,:], pred_len=60, intercept=True)  # overide pred with prediction based on yearly correlation 

		pred=cat_smooth(ts, pred,rate=0.5)    # change prediction so that it concatenates with existing time serieis smoothly 
		pred=np.array(np.round(pred),dtype='int')	

		## set zero
		# pred[zero_num>=30,:]*=0  
		pred[pred<0]=0


		# print informatio
		print("Yearly trend number: %d" %np.sum(year_pos))	
		print("Periodicity number: %d"  %np.sum(week_pos))
		'''
		print("Increasing linear trend number: %d" %np.sum(lin_pos_increase))
		print("Decreasing linear trend number: %d" %np.sum(lin_pos_decrease))
		'''

		return pred

	elif ts.ndim==1:
		pred=np.zeros(pred_len)
		ts=remove_na(ts)
		if len(ts)==0:	
			return pred
		print("The programmer is drinking coffee... ")
		raise NotImplementedError
	else:
		raise NotImplementedError 

'''
### Result Aggregation 
from .Submission import Scorer
## @param: test_data -- real result 
## @param: res=[res[1],...,res[n]] is the list of results from different models 
def aggregate(res,test_data,method="best"):
	n=len(res)
	final_res=np.array(res[1]*0,dtype='float64')      # initialization 
	if method=='average':
		for i in range(n):
			final_res+=res[i]
		return np.array(final_res/n, dtype='int')    # every res[i]>0 so we don't need to take care of negative case 
	elif method=='best':
		N=final_res.shape[1]  
		ave_score=np.zeros([N,n]) 
		scorer = Scorer(debug=True)
		for i in range(n):
			final,score_matrix = scorer.smape(res[i],test_data)
			ave_score[:,i] = score_matrix.mean(axis=1)
		idx= np.argmax(ave_score,axis=1)
		for i in range(N):
			final_res[i,:]= res[idx[i]][i,:]
		return np.array(final_res,dtype='int')  
'''





	