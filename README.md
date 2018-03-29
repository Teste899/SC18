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
```
python script.py -h
```

### Examples usage:
```
python power_monitor.py ‘1,2,4,8,10,12,14,16,18,20,22,24,26,28,30,32’ 32 60 30 power_model
python power_fitting.py power_model

python2.7 performance_monitor.py cpu_load '16,32' 32 '_nt_ 5,_nt_ 10' 1 10
python2.7 performance_fitting.py cpu_load 1

the argument with _nt_ will be replace by the number of treads on the executions
```


