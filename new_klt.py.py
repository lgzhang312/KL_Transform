import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d 
from scipy import linalg as LA 
from obspy import read 
from math import *


#================= read the original data =======================#
filename=read('nmo.segy')
tracls=len(filename)
input_data=np.asarray([np.copy(x) for x in filename.traces[0:tracls]])
plt.imshow(input_data.T,cmap='gray')
plt.title('Original data image')
plt.show()

#========================mute the data===================#

cut_data=input_data.T[150:len(input_data.T[:,0]),:]
plt.imshow(cut_data,cmap='gray')
plt.title('horizontal mute image')
plt.show()

row,column=cut_data.shape
print('the shape of the data is row=%d,column=%d' % cut_data.shape)
print('\n')
x1=int(input('Please enter the start point x1=(< %d):' % column)) 
print('\n')
y2=int(input('Please enter the end point y2=(< %d):' % row))
x2=column
k=y2/(x2-x1)

X=np.zeros(y2)
Y=np.zeros(y2)

for y in range(y2):
	x=int(x1+y/k)
	cut_data[y,x:column]=0
	Y[y]=y
	X[y]=x
plt.imshow(cut_data,cmap='gray')
plt.plot(X,Y,color='r',ls='--',label='mute line')
plt.legend(loc='upper right',fontsize='x-small')
plt.xlabel('Trace No')
plt.ylabel('Sample No')
plt.title('Mute data ')
plt.colorbar()
plt.savefig('mute_image.png',dpi=300)
plt.show()

#===================== stretch data ==================================#

last=np.zeros_like(cut_data)
Eachline_data_len=np.zeros(row)
re_data=np.zeros_like(last)

for j in range(row):
	line_data=cut_data[j,:]
	for i in range(1,column+1):
		if line_data[-i]!=0:
			count=column+1-i
			break
	Eachline_data_len[j]=count
	line_cut=line_data[0:count]
	x=np.linspace(0, count-1,num=count,endpoint=True)
	f1=interp1d(x, line_cut,kind='quadratic')
	xint=np.linspace(x.min(), x.max(),column)
	key=f1(xint)
	last[j,:]=key 

plt.imshow(last,cmap='gray')
plt.title('Stretch data ')
plt.xlabel('Trace No')
plt.ylabel('Sample No')
plt.colorbar()
plt.savefig('stretch.png',dpi=300)
plt.show()

#==========================re_stretch===========================#
def re_stretch(transfer,re_data):
	for z in range(row):
		new_line_data=transfer[z,:]
		re_len=Eachline_data_len[z]
		x_point=np.linspace(0, column,num=column,endpoint=True)
		f2=interp1d(x_point,new_line_data,kind='quadratic')
		x_int=np.linspace(0,column,num=re_len,endpoint=True)
		new=f2(x_int)
		re_data[z,0:int(re_len)]=new
	return re_data

re_stretch(last,re_data)
plt.imshow(re_data,cmap='gray')
plt.title('re_stretch data image')
plt.show()




key=last.T

#plt.imshow(key.T,cmap='gray')
#plt.show()
#==============================KL_Transform==================================#

eig_values,eig_vectors=LA.eig(np.cov(key))

idx=eig_values.argsort()[::-1]  #sort  in the decreasing
eig_values=np.real(eig_values[idx])
eig_vectors=np.real(eig_vectors[:,idx])

A=np.dot(eig_vectors.T,key)

number=int(input('How many eigenvalues you want to printf:'))
percent=np.zeros(number)
first=np.zeros((number,eig_vectors.shape[0],eig_vectors.shape[1]))
first_total=np.copy(eig_vectors)

for k in range(number):
	percent[k]=eig_values[k]/sum(eig_values)
	first[k,:,k]=eig_vectors[:,k]

loop=ceil(number/2)

counter=0
for i in range(loop):
	fig,axs=plt.subplots(1,2)
	for j,ax in enumerate(axs):
		im=ax.imshow(np.dot(first[counter], A).T,cmap='gray')
		ax.annotate("percent={:.2%} ".format(percent[counter]),xy=(0.05,0.1),\
					xycoords="axes fraction",color='b',fontsize=12)
		ax.set_xlabel('Trace No')
		ax.set_ylabel('Sample No')
		ax.set_title(str(counter+1)+' '+'eigenvalue')
		fig.colorbar(im,ax=ax)
		counter+=1
		if counter==number:
			break

	fig.tight_layout()
	fig.savefig('only'+str(counter-1)+','+str(counter)+'.png')
	plt.show()


first_total[:,number:column]=0

plt.imshow(np.dot(first_total, A).T,cmap='gray')
plt.title('Fisrt 4 eigenvalues')
plt.show()
'''



#========================plot fisrt 4 eigenvalues ========================#
plt.imshow(np.dot(first_1_4, A).T,cmap='gray')
plt.title('Fisrt 4 eigenvalues')
plt.annotate("percent={:.2%} ".format(sum4/100),xy=(0.05,0.1),\
	xycoords="axes fraction",color='b',fontsize=12)
plt.xlabel('Trace No')
plt.ylabel('Sample No')
plt.colorbar()
plt.savefig('Fisrt 4 eigenvalues.png',dpi=300)
plt.show()

#plt.clf()


#========================= re_stretch eigenvalues====================#

re_fourth=np.zeros_like(last)

re_stretch(np.dot(first_1_4, A).T, re_fourth)
plt.imshow(re_fourth,cmap='gray')
plt.title('de_stretch first 4 eigenvalues')
plt.xlabel('Trace No')
plt.ylabel('Sample No')
plt.tight_layout()
plt.colorbar()
plt.savefig('re_stretch 4.png',dpi=300)
plt.show()

re_stretch(np.dot(first_1,A),re_first)
re_stretch(np.dot(first_2,A),re_second)
re_stretch(np.dot(first_3,A),re_third)
re_stretch(np.dot(first_4,A),re_fourth)


fig,ax=plt.subplots(1,2)
ax[0].imshow(re_first,cmap='gray')
ax[0].set_xlabel('Trace No')
ax[0].set_ylabel('Sample No')
ax[0].set_title(' 1st eigenvalue',fontsize=10)
ax[1].imshow(re_second,cmap='gray')
ax[1].set_xlabel('Trace No')
ax[1].set_ylabel('Sample No')
ax[1].set_title(' 2nd eigenvalue',fontsize=10)

fig.tight_layout()
plt.show()
fig.savefig('re_stretch first 1,2 eigenvalues.png',dpi=300)

pic,ax=plt.subplots(1,2)
ax[0].imshow(re_third,cmap='gray')
ax[0].set_xlabel('Trace No')
ax[0].set_ylabel('Sample No')
ax[0].set_title(' 3rd eigenvalue',fontsize=10)
ax[1].imshow(re_fourth,cmap='gray')
ax[1].set_xlabel('Trace No')
ax[1].set_ylabel('Sample No')
ax[1].set_title(' 4th eigenvalue',fontsize=10)

pic.tight_layout()
plt.show()
pic.savefig('re_stretch first 3,4 eigenvalues.png',dpi=300)


#plt.imshow(re_third,cmap='gray')
#plt.show()
'''