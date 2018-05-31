# SC18

## Requeriments
* Linux kernel with root acess
* acpi-cpufreq driver
* IPMI


## Scripts description

* power_monitor: collect power information using IPMI and a program to stress the cpu to the maximum.
* power_fitting: calculates the power model constants of the system and compute the errors using data from the power_monitor.
* performance_monitor: makes the complete modeling from the collection of the times of execution to the calculation of the SVR.
* performance_fitting: use the SVR model and the power model to compute the energy and find the minimal energy configuration.

## Execution help

Each script has its own parameters that are detailed using -h option

```
python script.py -h
```

## Environment setup
```
source sc18/bin/activate
```

### Examples usage:
```
python power_monitor.py ‘1,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32’ 32 60 30 power_model
python power_fitting.py power_model

python2.7 performance_monitor.py cpu_load '16,32' 32 '_nt_ 5,_nt_ 10' 1 10
python2.7 performance_fitting.py cpu_load 1

the argument with _nt_ will be replace by the number of treads on the executions
```

Examples of input arguments used on some applications:

#### PARSEC Blackscholes

'_nt_', 'in_625K.txt', 'out_625K.txt'

'_nt_', 'in_1M.txt', 'out_1M.txt'

'_nt_', 'in_2M.txt', 'out_2M.txt'

'_nt_', 'in_5M.txt', 'out_5M.txt'

'_nt_', 'in_10M.txt', 'out_10M.txt'


#### PARSEC Fluidanimate

'_nt_', '250', 'in_500K.fluid'

'_nt_', '500', 'in_500K.fluid'

'_nt_', '1000', 'in_500K.fluid'

'_nt_', '2000', 'in_500K.fluid'

'_nt_', '4000', 'in_500K.fluid'

#### PARSEC Raytrace

'thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '140', '140'

'thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '196', '196'

'thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '274', '274'

'thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '384', '384'

'thai_statue.obj','-automove', '-nthreads', '_nt_', '-frames', '10000', '-res', '537', '537'

#### PARSEC Swaptions

'-ns', '64', '-sm', '3000000', '-nt', '_nt_'
'-ns', '64', '-sm', '4000000', '-nt', '_nt_'
'-ns', '64', '-sm', '5000000', '-nt', '_nt_'
'-ns', '64', '-sm', '6000000', '-nt', '_nt_'
'-ns', '64', '-sm', '7000000', '-nt', '_nt_'

The PARSEC 3.0 Benchmark and its necessary input files are available at http://parsec.cs.princeton.edu/download.htm.

## Execution order
The monitoring scripts must first be run to collect the necessary data for be used by fitting scripts. A execution example of the power script is:

### Power scripts

```
python power_monitor.py 1,2,4,8,10,12 16 60 30 power_model
python power_fitting.py power_model
```

This script will collect power information and to perform the model fitting and its validation.

### Performance scripts
```
python2.7 performance_monitor.py cpu_load "16,32" 32 "nt_ 5,_nt_ 10" 1 10
python2.7 performance_fitting.py cpu_load 1
```

The performance scripts will also collect data of the power consumption and execution time. From the execution time it is
going to calculate the SVR estimation and perform a cross validation. The energy equation its also calculated there using
the power model previously calculated and all the results can be dynamically observed.
