import ipmi
import cpufreq

import subprocess
import threading
import pickle
import time
import os
import argparse

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score


def load_data(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_data(name, data_):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data_, f, pickle.HIGHEST_PROTOCOL)

class programa(threading.Thread):
	def __init__(self, program_, args_):
		threading.Thread.__init__(self)
		self.args= args_
		self.program= program_

	def run(self):
		print ["./"+str(self.program)]+self.args
		subprocess.call(["./"+str(self.program)]+self.args)

def monitoring(program_name, list_threads, max_threads, list_args, idle_time, save_name):

	sensor= ipmi.IPMI('http://localhost:8080', user='admin', password= 'admin')
	cpu= cpufreq.CPUFreq()

	for xcpu in range(1,max_threads):
		cpu.enable_cpu(xcpu)
	cpu.change_governo("userspace")
	freq= cpu.get_frequencies()[0]['data']
#	for xcpu in range(32,64):
#                cpu.disable_cpu(xcpu)
	models= []
	for f in freq[:2]:
		for xcpu in range(0,max_threads):
			cpu.change_frequency(f, xcpu)
		try:
			info_threads= []
			for thr in list_threads:
				info_pcpu= []
				for arg in list_args:
					for xcpu in range(thr,max_threads):
						cpu.disable_cpu(xcpu)
					arg= map(str, arg.split(' '))
					arg= map(lambda s: str.replace(s, '_nt_', str(thr)), arg)
					print 'Argument ', arg
					program= programa(program_name,arg)
					program.start()
					info_sensor= []
					ti = time.time()
					while program.is_alive():
						print 'Time ', time.time()-ti, "Frequency ", f, " nThreads", thr
						tg= time.time()
						info= {'time': time.time()-ti,'sensor':sensor.get_data()}
						info_sensor.append(info.copy())
						print 'Power'
						for i, font in enumerate(info['sensor']['sources']):
							print 'Source ', i , font['dcOutPower'], 'W'
						tg= time.time()-tg
						if 1-tg >= 0:
							time.sleep(1-tg)
					tt= time.time()-ti
					program.join()
					print 'Total time ', tt
					l1= {'arg':list(arg), 'total_time':tt, 'ipmi':info_sensor}
					info_pcpu.append(l1.copy())
					for xcpu in range(1,max_threads):
						cpu.enable_cpu(xcpu)
					time.sleep(idle_time)
					for xcpu in range(0,max_threads):
						cpu.change_frequency(f, xcpu)
					time.sleep(idle_time)
				l2= {'nthread':thr, 'lpcpu':info_pcpu}
				info_threads.append(l2.copy())
		except Exception as e:
			print e
		model= {'freq':f, 'threads':info_threads}
		models.append(model.copy())
	
	f= open(save_name+'.pkl', 'wb')
	pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

	for xcpu in range(1,64):
		cpu.enable_cpu(xcpu)
	cpu.change_governo("userspace")

	return models


def create_df(data, output, arg_num):
    dados = []
    for d in data:
        for t in d['threads']:
            for p in t['lpcpu']:
                pw = []
                energia = 0
                for i in range(len(p['ipmi'])):
                    pot = p['ipmi'][i]['sensor']['sources'][0]['dcOutPower']
                    pot += p['ipmi'][i]['sensor']['sources'][1]['dcOutPower']
                    if i - 1 >= 0:
                        energia += (p['ipmi'][i]['time'] - p['ipmi'][i - 1]['time']) * pot
                    if i == len(p['ipmi']) - 1:
                        energia += pot * (p['total_time'] - p['ipmi'][i]['time'])

                for s in p['ipmi']:
                    pot = float(s['sensor']['sources'][0]['dcOutPower'] + s['sensor']['sources'][1]['dcOutPower'])
                    pw.append(pot)
                pw = np.asarray(pw)
                row = [d['freq'], t['nthread'], p['arg'][arg_num], p['total_time'], pw.mean()]
                dados.append(row)
    df= pd.DataFrame(dados, columns=['freq', 'thr', 'in', 'time', 'pw'])

    df['energy'] = df['time'] * df['pw']

    cat = pd.factorize(df['in'])
    df['in'] = cat[0] + 1
    df['freq'] = df['freq'].astype(float) / 1e6

    save_data(output, df)
    save_data(output + '_cat', cat)

    return df, cat

def characterize(name, df):
	X = df[['freq', 'thr', 'in']].values
	Y = df['time'].astype(float).values

	clf = SVR(kernel='rbf', C=10e3, gamma=0.5)
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1, random_state=0)

	print clf.fit(Xtrain, Ytrain)

	save_data(name+'_time', clf)

	return clf, 0


def validate(df, clf):
    X = df[['freq', 'thr', 'in']].values
    Y = df['time'].astype(float).values

    scores = cross_val_score(clf, X, Y, cv=10,
                             scoring=lambda clf, X, y: np.sum(np.abs(y - clf.predict(X)) / y) / len(y))
    print 'Cross validation mpe', scores, np.mean(scores)

    scores = cross_val_score(clf, X, Y, cv=10,
                             scoring=lambda clf, X, y: np.sum(np.abs(y - clf.predict(X))) / len(y))
    print 'Cross validation mae', scores, np.mean(scores)


args_black= [['_nt_', 'in_312K.txt', 'out_312K.txt'],\
	['_nt_', 'in_625K.txt', 'out_625K.txt'],\
	['_nt_', 'in_1M.txt', 'out_1M.txt'],\
	['_nt_', 'in_2M.txt', 'out_2M.txt'],\
	['_nt_', 'in_5M.txt', 'out_5M.txt'],\
	['_nt_', 'in_10M.txt', 'out_10M.txt']]

args_fluid= [['_nt_', '125', 'in_500K.fluid'],\
	['_nt_', '250', 'in_500K.fluid'],\
	['_nt_', '500', 'in_500K.fluid'],\
	['_nt_', '1000', 'in_500K.fluid'],\
	['_nt_', '2000', 'in_500K.fluid'],\
	['_nt_', '4000', 'in_500K.fluid']]

args_rtview= [['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '100', '100'],\
	['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '140', '140'],\
	['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '196', '196'],\
	['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '274', '274'],\
	['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '384', '384'],\
	['thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '537', '537']]

args_swap= [['-ns', '64', '-sm', '2000000', '-nt', '_nt_'],\
	['-ns', '64', '-sm', '3000000', '-nt', '_nt_'],\
	['-ns', '64', '-sm', '4000000', '-nt', '_nt_'],\
	['-ns', '64', '-sm', '5000000', '-nt', '_nt_'],\
	['-ns', '64', '-sm', '6000000', '-nt', '_nt_'],\
	['-ns', '64', '-sm', '7000000', '-nt', '_nt_']]

parser = argparse.ArgumentParser(description='Power model fitting')
parser.add_argument('programname', type=str, help='the program to perform')
parser.add_argument('list_of_threads', type=str, help='list of threads separeted by comma')
parser.add_argument('num_of_cores', type=int, help='total number of cores')
parser.add_argument('list_of_args', type=str, help='list of arguments separeted by comma')
parser.add_argument('input_arg', type=int, help='argument that is the input size')
parser.add_argument('idle_time', type=int, help='idle time between executions')

args = parser.parse_args()

data= monitoring(program_name=args.programname, list_threads= map(int, args.list_of_threads.split(',')), max_threads= args.num_of_cores, list_args= map(str, args.list_of_args.split(',')), idle_time= args.idle_time, save_name= args.programname)

#fluid_data= monitoring(program_name='fluidanimate',list_threads= [1,2,4,8,16,32], list_args= args_fluid, idle_time= 30, save_name='raw_fluid_data')
#monitoring(program_name='rtview',list_threads= [1]+[2*x for x in range(1,17)], list_args= args_rtview, idle_time= 30, save_name='raw_rtview_data')
#monitoring(program_name='swaptions',list_threads= [1]+[2*x for x in range(1,17)], list_args= args_swap, idle_time= 30, save_name='raw_swap')
#monitoring(program_name='blackscholes',list_threads= [1]+[2*x for x in range(1,17)], list_args= args_black, idle_time= 30, save_name='raw_black')

data, cat = create_df(data= data, output= args.programname+'_performance', arg_num= args.input_arg)
svm_time, svm_pw= characterize(args.programname+'_carac', data)
validate(data,svm_time)
