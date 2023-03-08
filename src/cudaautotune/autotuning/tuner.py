import subprocess
import argparse

import opentuner
from opentuner.measurement import MeasurementInterface
from opentuner.search.manipulator import (ConfigurationManipulator,
                                          IntegerParameter)

parser = argparse.ArgumentParser(parents=opentuner.argparsers())

base_path = "/home/nfs/asubah/dev/pixeltrack-standalone/"

# Heating up the GPU before tuning
process_path = base_path + "cudaautotune"

print('Preheating started')
cmd = [process_path, "--runForMinutes", "10", "--numberOfThreads", "12", "--numberOfStreams", "8"]
subprocess.run(cmd)
print('Preheating finished')

class CMSSWTuner(MeasurementInterface):
    def output_config_file(self, params):
        from mako.template import Template
        template = Template(filename=base_path+"src/cudaautotune/autotuning/configs.mako")
        with open(base_path+"src/cudaautotune/autotuning/configs", "w") as f:
            f.write(template.render(**params))
    
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        # manipulator.add_parameter(IntegerParameter("numberOfThreads", 1, 24))
        # manipulator.add_parameter(IntegerParameter("numberOfStreams", 1, 24))
        manipulator.add_parameter(IntegerParameter("findClus", 1, 32))
        manipulator.add_parameter(IntegerParameter("RawToDigi_kernel", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_connect_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_connect_stride", 1, 8))
        manipulator.add_parameter(IntegerParameter("getHits", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_find_ntuplets", 1, 32))
        manipulator.add_parameter(IntegerParameter("fishbone_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("fishbone_stride", 1, 32))
        manipulator.add_parameter(IntegerParameter("clusterChargeCut", 1, 32))
        manipulator.add_parameter(IntegerParameter("calibDigis", 1, 32))
        manipulator.add_parameter(IntegerParameter("countModules", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit3_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit4_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit5_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit3_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit4_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelLineFit5_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit3_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit4_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit5_threads", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit3_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit4_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernelFastFit5_blocks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_fillHitDetIndices", 1, 32))
        manipulator.add_parameter(IntegerParameter("finalizeBulk", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_earlyDuplicateRemover", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_countMultiplicity", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_fillMultiplicity", 1, 32))
        manipulator.add_parameter(IntegerParameter("initDoublets", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_classifyTracks", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_fishboneCleaner", 1, 32))
        manipulator.add_parameter(IntegerParameter("kernel_fastDuplicateRemover", 1, 32))

        return manipulator
    
    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        self.output_config_file(cfg)
    
        # Benchmarking
        time = float('inf')
        throughput = float('-inf')
        cpu_efficiency = ""
        status = ""
        validation = ""
        output = ""
        
        cmd = [process_path,
               "--numberOfThreads", str(12),# str(cfg["numberOfThreads"]),
               "--numberOfStreams", str(8),# str(cfg["numberOfStreams"]),
               "--validation"]
        
        # print('Validation command:', cmd)
        process = subprocess.run(cmd, encoding='UTF-8', capture_output=True)
        if (process.returncode == 0):
            # print('Validation output:\n', process.stdout)
            if process.stdout.find("passed validation"):
                validation = "PASSED"
                status = "OK"
                # print(process.stdout)
                output = process.stdout.split('\n')[-2].split(' ')
                # print(output)
                time = float(output[4])
                throughput = output[7]
                cpu_efficiency = output[13]
            elif process.stdout.find("failed validation"):
                validation = "FAILED"
            else:
                validation = "ERROR"
        else:
            # print(process.stdout)
            validation = "ERROR"
            time = float('inf')
        
        # print('Validation result:', validation)
    
        cfg["throughput"] = throughput
        cfg["cpu_efficiency"] = cpu_efficiency
        cfg["status"] = status
        cfg["validation"] = validation
        cfg["time"] = time

        # print(cfg)
    
        return opentuner.resultsdb.models.Result(time=time)

if __name__ == '__main__':
    args = parser.parse_args()
    CMSSWTuner.main(args)
