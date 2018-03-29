import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.optimize import least_squares
import os
import argparse

def plot3D(x, y, z, label, pontos= False):
    X = np.arange(0, len(x), 1)
    Y = np.arange(0, len(y), 1)
    X, Y = np.meshgrid(X, Y)
    Z = z[X, Y]

    if pontos:
        surf = ax.scatter(X, Y, Z, antialiased=True, color='k')
    else:
        surf = ax.plot_wireframe(X, Y, Z,antialiased=True,color='k')

    ax.set_xlabel('Frequency (GHz)', fontsize=10)
    ax.set_xticks(np.arange(0, len(x), 1))
    ax.set_xticklabels(x, fontsize=10)

    ax.set_ylabel('Active cores', fontsize=10)
    ax.set_yticks(np.arange(0, len(y), 1))
    ax.set_yticklabels(y, fontsize=10)

    ax.set_zlabel(label, fontsize=10)

def reject_offcurve(data, m = 2.):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else 0.
	if mdev == 0:
		return data
	return data[s<m]

def getVectorIPMI(ipmi):
	ret1 = []
	ret2 = []
	for t in ipmi:
		ret1.append(t['sources'][0]['dcOutPower'])
		ret2.append(t['sources'][1]['dcOutPower'])
	ret1 = np.sort(ret1)
	ret2 = np.sort(ret2)
	return reject_offcurve(ret1), reject_offcurve(ret2)

def processData(data, verbose= False):
	frs = []
	thrs = []
	pws = []

	if verbose:
		print "Samples Frequency nThreads Power std"

	for d in data[:]:
		frs.append(float(d['freq'])/1e6)
		for thr in d['threads'][:]:
			for pcpu in thr['lpcpu']:
				pw=pw_std= 0
				pw_1, pw_2 = getVectorIPMI(pcpu['ipmi'])
				if len(pw_2) > 0:
					pw += np.mean(pw_1)+np.mean(pw_2)
					pw_std += np.std(np.concatenate((pw_1, pw_1), axis=0))
				else:
					pw += np.mean(pw_1)
					pw_std += np.std(np.array(pw_1))
				
				if verbose: print '{:.2f} {} {} {:.2f}+-{:.2f}'.format(len(pw_1), d['freq'], thr['nthread'], pw, pw_std)
				if verbose: print '-' * 30

				pws.append(pw)

	for thr in data[0]['threads'][:]:
		thrs.append(thr['nthread'])

	return frs, thrs, pws


def objective_power_fun(x, f, p, y):
    return x[0]*f**3*p+x[1]*f*p+x[2]+x[3]*(np.floor(p/17)+1)-y

def power_fun(x, f, p):
    return x[0]*f**3*p+x[1]*f*p+x[2]+x[3]*(np.floor(p/17)+1)

def fitting(frs, thrs, pws):
	x0= np.ones(4)
	f=[]
	p=[]
	for f_ in frs:
		for n_ in thrs:
		    f.append(f_)
		    p.append(n_)
	pws= pws.reshape(-1)
	f= np.asarray(f,dtype=float)
	p= np.asarray(p,dtype=float)
	res_robust= least_squares(objective_power_fun, x0, loss='soft_l1', f_scale=0.1, args=(f, p, pws))

	return res_robust.x

parser = argparse.ArgumentParser(description='Power model fitting')
parser.add_argument('filename', type=str, help='data collected from monitoring script')

args = parser.parse_args()

# load and process data collected
f = open(args.filename, 'rb+')
data = pickle.load(f)
frs, thrs, pws = processData(data= data, verbose= True)

# create the figure and display power measurements
fig = plt.figure()
ax = fig.gca(projection='3d')

pws = np.reshape(pws, (len(frs), len(thrs)))
plot3D(frs, thrs, pws, 'Power (W)', True)

# Fit the model equations with the data
x0 = fitting(frs, thrs, pws)
print 'Constantes values', x0

# Calculate the model surface and display
y_model= []
for a in frs:
    for b in thrs:
        y_model.append(power_fun(x0, a, b))

y_model = np.reshape(y_model, (len(frs), len(thrs)))
plot3D(frs, thrs, y_model, 'Power (W)')

# Calculate model erros
error= np.sum((np.abs(y_model.reshape(-1)-pws.reshape(-1))))/y_model.size
print 'Absolut error', error
error= np.mean((np.abs(y_model.reshape(-1)-pws.reshape(-1))/pws.reshape(-1)))
print 'Percentual error', error*100
error= np.mean(((y_model.reshape(-1)-pws.reshape(-1))**2))
print 'Root square error', error

plt.legend(['measurements', 'model'])
plt.tight_layout()
plt.show()
