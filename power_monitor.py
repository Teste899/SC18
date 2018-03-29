import ipmi
import cpufreq

import subprocess
import threading
import pickle
import time
import os
import argparse

class CPUload(threading.Thread):
	def __init__(self, nt_, time_):
		threading.Thread.__init__(self)
		self.nt= nt_
		self.time= time_

	def run(self):
		subprocess.call(['./cpu_load', self.nt, self.time])

def monitoring(list_threads, max_threads, list_pcpu, busy_time, idle_time, output_file):
	
	try:
		sensor= ipmi.IPMI('http://localhost:8080', user='admin', password= 'admin')
		cpu= cpufreq.CPUFreq()

		for xcpu in range(1,max_threads):
			cpu.enable_cpu(xcpu) 
		cpu.change_governo("userspace")

		freq= cpu.get_frequencies()[0]['data'] # get all available frequencies for cpu0 and assume that is the same for the others
		models= []
		for f in freq[:2]:
			cpu.change_frequency(f)
			try:
				info_threads= []
				for thr in list_threads:
					info_pcpu= []
					for pcpu in list_pcpu:
						for xcpu in range(thr,max_threads):
							cpu.disable_cpu(xcpu)
						program= CPUload(str(thr), str(busy_time))
						program.start()
						info_sensor= []
						while program.is_alive():
							info= sensor.get_data()
							info_sensor.append(info.copy())
							print 'Frequency', f, 'threads', thr
							print 'Power'
							for i, font in enumerate(info['sources']):
								print 'Source ', i , font['dcOutPower'], 'W'
							#sensor.print_data()
						program.join()
						l1= {'pcpu':pcpu, 'ipmi':info_sensor}
						info_pcpu.append(l1.copy())
						for xcpu in range(1,max_threads):
							cpu.enable_cpu(xcpu)
						time.sleep(idle_time)
						cpu.change_frequency(f)
					l2= {'nthread':thr, 'lpcpu':info_pcpu}
					info_threads.append(l2.copy())
			except Exception as e:
				print e
			model= {'freq':f, 'threads':info_threads}
			models.append(model.copy())

		with open(output_file+'.pkl', 'wb') as f:
			pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)
		cpu.change_governo("power_save")

	except Exception as e:
		print e

parser = argparse.ArgumentParser(description='Power model fitting')
parser.add_argument('list_of_threads', type=str, help='list of threads separeted by comma')
parser.add_argument('num_of_cores', type=int, help='total number of cores')
parser.add_argument('execution_time', type=int, help='time of each execution')
parser.add_argument('idle_time', type=int, help='idle time between executions')
parser.add_argument('output_file', type=str, help='name of the output file with the data collected')

args = parser.parse_args()

monitoring(list_threads= map(int, args.list_of_threads.split(',')), max_threads= args.num_of_cores, list_pcpu= [1], busy_time= args.execution_time, idle_time= args.idle_time, output_file= args.output_file)
