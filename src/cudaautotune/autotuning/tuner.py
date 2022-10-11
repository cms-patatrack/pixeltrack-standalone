import argparse
import pathlib
import subprocess
import itertools
import random
import datetime

parser = argparse.ArgumentParser(description='Autotuner script.')
parser.add_argument('-p', '--process', type=pathlib.Path, nargs=1, required=True,
        help='path to the program to be autotuned')
parser.add_argument('-c', '--configurations', type=pathlib.Path, nargs=1,
        default=[pathlib.Path('src/cudaautotune/autotuning/kernel_configs')], help='path to save the configurations for the tunable process to read them. Default = autotuning/kernel_configs/')
parser.add_argument('-t', '--tunables', type=argparse.FileType('r'), nargs=1,
        default=[open('src/cudaautotune/autotuning/tunables.csv', 'r')], help='csv file that contains the tunable parameters. Default = autotuning/tunables.csv')
parser.add_argument('-r', '--results', type=argparse.FileType('a'), nargs=1,
        default=[open('src/cudaautotune/autotuning/results.txt', 'a')],
        help='results file, if it exists, results will be appended to the end, if not, it will be created. Default = autotuning/results.txt')
parser.add_argument('--validation', nargs='?', const=True, default=False,
        help='validate the output of the process. Default = False')
parser.add_argument('--preheat',  nargs='?', const=True, default=False,
        help='run the process multiple times before autotuning. Default = False')
parser.add_argument('--verbose', '-v', action='count', default=0)

args = parser.parse_args()

tunables = {}
tunables_list = []
lbounds = []
ubounds = []
steps = []

tunables_file = args.tunables[0]
for n, line in enumerate(tunables_file.readlines()):
    parameters = line.strip('\n').split(',')

    if args.verbose > 1:
        print(parameters)

    tunables[parameters[0]] = n
    tunables_list.append(parameters[0])
    lbounds.append(int(parameters[1]))
    ubounds.append(int(parameters[2]))
    steps.append(int(parameters[3]))

tunables_file.close()
 
configurations = []
ranges = [range(l, u + 1, s) for l, u, s in zip(lbounds, ubounds, steps)]
length = 1
for r in ranges:
    r = list(r)
    length = length * len(r)
    random.shuffle(r)
    configurations.append(r)

if args.verbose:
    print("number of configurations is " + str(length))

# Heating up the GPU before tuning
process_path = args.process[0].resolve()

if args.preheat:
    if args.verbose:
        print('Preheating started')
    cmd = [process_path, "--runForMinutes", "1", "--numberOfThreads", "12", "--numberOfStreams", "12"]
    if args.verbose > 1:
        print('Preheat command:', cmd)
    subprocess.run(cmd)
    if args.verbose:
        print('Preheating finished')

# Tuning
configurations = list(itertools.product(*configurations))
random.shuffle(configurations)
results_file = args.results[0]
kernels_config_path = args.configurations[0]
kernels = tunables_list[:-2]
for config in configurations:
    if args.verbose:
        print('Configuration:', config)

    # TODO write only when the parameter change
    for kernel in kernels:
        file = open(kernels_config_path / kernel, 'w')
        file.write(str(config[tunables[kernel]]))
        file.close()
    
    # Validation
    validation = ""
    cmd = [process_path,
            "--numberOfThreads", "12",
            "--numberOfStreams", "12",
            "--validation"]
    if args.verbose > 1:
        print('Validation command:', cmd)
    process = subprocess.run(cmd, encoding='UTF-8', capture_output=True)
    if (process.returncode == 0):
        if args.verbose > 1:
            print('Validation output:', process.stdout)
        if process.stdout.find("passed validation"):
            validation = "PASSED"
        elif process.stdout.find("failed validation"):
            validation = "FAILED"
        else:
            validation = "ERROR"
    else:
        print(process.stdout)
        validation = "ERROR"
    if args.verbose:
        print('Validation result:', validation)
        
    # Benchmarking
    time = "NaN"
    throughput = "NaN"
    cpu_efficiency = "NaN"
    status = ""

    cpu_threads = config[tunables["cpu_threads"]]
    gpu_streams = cpu_threads + config[tunables["gpu_streams"]]
    cmd = [process_path,
            "--maxEvents", "10000",
            "--numberOfThreads", str(cpu_threads),
            "--numberOfStreams", str(gpu_streams)]
    if args.verbose > 1:
        print('Tuning command:', cmd)
    process = subprocess.run(cmd, encoding='UTF-8', capture_output=True)

    output = process.stdout
    if (process.returncode == 0):
        if args.verbose > 1:
            print(output)
        status = "OK"
        output = output.split('\n')[2].split(' ')
        time = output[4]
        throughput = output[7]
        cpu_efficiency = output[13]
    else:
        print(output)
        status = "ERROR"
        
    result = ' '.join([str(datetime.datetime.now()),
        throughput, time, cpu_efficiency, status, validation, str(config)]) + '\n'
    results_file.write(result)
    results_file.flush()

    if args.verbose:
        print('Result:', result)

results_file.close()
