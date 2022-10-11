# Autotuning Script
_Requires Python 3.7 or Higher_

A python script that can be used to autotune parameters in the pixeltrack-standalone project. The parameters can be anything as long as there is a n interface in the program that can read the parameters from the configurations folder. Hence, an interface was created to autotune CUDA kernel launch configurations

[`src/cudaautotune/CUDACore/ExecutionConfiguration.h`](../CUDACore/ExecutionConfiguration.h)

**Example:**
```c++
#include "CUDACore/ExecutionConfiguration.h"

cms::cuda::ExecutionConfiguration exec;
int threadsPerBlock = exec.configFromFile("RawToDigi_kernel");

int threadsPerBlock = exec.configFromFile("calibDigis");// 256;
int blocks = (std::max(int(wordCounter), int(gpuClustering::MaxNumModules)) + threadsPerBlock - 1) / threadsPerBlock;

gpuCalibPixel::calibDigis<<<blocks, threadsPerBlock, 0, stream>>>(...)
...
```

Other parameters can be tuned also such as number of CPU threads and GPU streams.

## Usage

```bash
usage: tuner.py [-h] -p PROCESS [-c CONFIGURATIONS] [-t TUNABLES] [-r RESULTS] [--validation [VALIDATION]]
                [--preheat [PREHEAT]] [--verbose]

Autotuner script.

options:
  -h, --help            show this help message and exit
  -p PROCESS, --process PROCESS
                        path to the program to be autotuned
  -c CONFIGURATIONS, --configurations CONFIGURATIONS
                        path to save the configurations for the tunable process to read them. Default =
                        src/cudaautotune/autotuning/kernel_configs/
  -t TUNABLES, --tunables TUNABLES
                        csv file that contains the tunable parameters. Default = src/cudaautotune/autotuning/tunables.csv
  -r RESULTS, --results RESULTS
                        results file, if it exists, results will be appended to the end, if not, it will be created.
                        Default = src/cudaautotune/autotuning/results.txt
  --validation [VALIDATION]
                        validate the output of the process. Default = False
  --preheat [PREHEAT]   run the process multiple times before autotuning. Default = False
  --verbose, -v
```

Simple run from the project root path:
```bash
python autotuning/tuner.py --process ./cudaautotune
```

You need to provide a `csv` file that contains the tunable parameters as follows:
```
prarmeter_name,lower_bound,upped_bound,incremental_step
```

Example file can be found here:

[`src/cudaautotune/autotuning/tunables.csv`](tunables.csv)

Example from results file:

```bash
2022-10-04 14:01:55.184845 1282.89 7.794910e+00 59.5% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 64, 192, 128, 64, 128, 256, 12, 4)
2022-10-04 14:02:07.028675 1281.11 7.805726e+00 60.8% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 128, 256, 128, 192, 128, 256, 12, 4)
2022-10-04 14:02:23.100261 1279.51 7.815468e+00 60.6% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 128, 256, 64, 64, 192, 256, 12, 4)
2022-10-04 14:02:35.089202 1272.24 7.860135e+00 58.4% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 64, 192, 256, 64, 192, 192, 12, 4)
2022-10-04 15:09:38.377397 1293.88 7.728671e+00 59.8% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 192, 256, 128, 64, 256, 192, 12, 4)
2022-10-04 15:10:02.501392 1284.46 7.785401e+00 60.2% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 64, 64, 256, 256, 128, 64, 12, 4)
2022-10-04 15:10:14.319484 1280.77 7.807828e+00 60.1% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 256, 256, 256, 192, 128, 128, 12, 4)
2022-10-04 15:20:05.433228 1289.93 7.752346e+00 60.2% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 64, 128, 192, 192, 256, 64, 12, 4)
2022-10-04 15:20:17.282590 1287.91 7.764510e+00 60.8% OK PASSED (128, 128, 64, 128, 128, 64, 64, 128, 128, 64, 256, 256, 128, 64, 64, 128, 256, 128, 192, 256, 256, 128, 64, 256, 192, 192, 64, 192, 256, 192, 128, 12, 4)
```
The columns are:
- Date
- Time
- Throughput (Parsed from output)
- Runtime (Parsed from output)
- CPU usage per thread (Parsed from output)
- Tunning process return (OK/ERROR)
- Valudation process return (PASSED/FAILED/ERROR)
- Configuration
