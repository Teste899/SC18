import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
import argparse

fig = plt.figure()
ax = fig.gca(projection='3d')

def plot3d(x, y, z, label, label_x, label_y, color_='b', pontos=False, axis=ax):
    X = np.arange(0, len(x), 1)
    Y = np.arange(0, len(y), 1)
    X, Y = np.meshgrid(X, Y)
    Z = z[X, Y]

    if pontos:
        axis.scatter(X, Y, Z, color=color_)
    else:
        axis.plot_wireframe(X, Y, Z, antialiased=True, color=color_)

    axis.set_xlabel(label_x, fontsize=10)
    axis.set_xticks(np.arange(0, len(x), 1))
    axis.set_xticklabels(x, fontsize=10)

    axis.set_ylabel(label_y, fontsize=10)
    axis.set_yticks(np.arange(0, len(y), 1))
    axis.set_yticklabels(y, fontsize=10)
    axis.set_zlabel(label, fontsize=10)

def power_model(f,n):
    return 0.29045563*f**3*n+1.97006881*f*n+207.77778443+9.18386954*int(n/16)

def load_data(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_data(name, data_):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data_, f, pickle.HIGHEST_PROTOCOL)

parser = argparse.ArgumentParser(description='Power model fitting')
parser.add_argument('filename', type=str, help='application name')
parser.add_argument('energy_time', type=int, help='bollean indicanting if the output is the energy or time')

args = parser.parse_args()

mostarEnergia = args.energy_time
model_data = pd.read_pickle(args.filename+'_performance.pkl')
categories = load_data(args.filename+'_performance_cat')
svm_time = load_data(args.filename+'_carac_time')

model_data= model_data.sort_values(['freq','thr'])

frs = model_data['freq'].astype(float).unique()
thrs = model_data['thr'].unique()
inps = model_data['in'].unique()

axInput = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor='lightgoldenrodyellow')
sin = Slider(axInput, 'input', model_data['in'].min(), model_data['in'].max(), valinit=1.0)


def update_data(val):
	global ax, sin, model_data, frs, thrs
	ax.clear()
	d = int(val / 0.1) * 0.1

	sin.val = d
	sin.poly.xy[2] = sin.val, 1
	sin.poly.xy[3] = sin.val, 0
	sin.valtext.set_text(sin.valfmt % sin.val)
	label_= 'Energy (kJ)' if mostarEnergia else 'Time (s)'

	pws = []
	dataBase = []
	for f in frs:
		for t in thrs:
		    if mostarEnergia:
		        pws.append(svm_time.predict([[f, t, d]])[0] * power_model(f, t) / 1000.0)
		    else:
		        pws.append(svm_time.predict([[f, t, d]])[0])
		    dataBase.append([f, t, d, pws[-1]])
	dataBase = pd.DataFrame(dataBase, columns=['freq', 'thr', 'in', 'energy2'])

	pws = np.reshape(pws, (len(frs), len(thrs)))
	
	minval = dataBase.sort_values('energy2').iloc[0, :]
	print 'Minimal configuration energy {} cores {} (GHz) freq.'.format(minval['thr'], minval['freq'])
	plot3d(frs, thrs, pws, label_, 'Frequency (GHz)', 'Active cores', 'k')
	if not model_data[model_data['in'] == d].empty:

		#print minval['energy2'] * 1000
		#print model_data[(model_data['in'] == d) &
		#                 (model_data['thr'] == minval['thr']) &
		#                 (model_data['freq'] == minval['freq'])][['freq', 'thr', 'in', 'energy']]
		# print dataBase.sort_values('energy2').head(5)
		# print data[data['in'] == d].sort_values('energy')[['freq', 'thr', 'in', 'energy']]

		if mostarEnergia:
		    pws = model_data[model_data['in'] == d]['energy'].values / 1000.0
		else:
		    pws = model_data[model_data['in'] == d]['time'].values

		pws = np.reshape(pws, (len(frs), len(thrs)))
		plot3d(frs, thrs, pws, label_, 'Frequency (GHz)', 'Active cores', color_='k', pontos=True)

	ax.set_title(categories[1][int(d) - 1])
	ax.legend(['Surface', 'Real points'])
	fig.canvas.draw_idle()


update_data(1.0)
sin.on_changed(update_data)

plt.show()
