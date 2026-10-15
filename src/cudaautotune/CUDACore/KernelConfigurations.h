#ifndef CUDACore_KernelConfigurations_h
#define CUDACore_KernelConfigurations_h

#include <fstream>
#include <iostream>
#include <unordered_map>

namespace cms {
  // using LaunchConfigs = std::unordered_map<std::string, int>;
  // class LaunchConfigs {
  //   public:
  // LaunchConfigs getLaunchConfigs(){
  std::unordered_map<std::string, int> getLaunchConfigs(){
    const std::string CONFIG_FILE = "src/cudaautotune/autotuning/configs";

    std::fstream file;
    file.open(CONFIG_FILE);

    // LaunchConfigs configurations;
    std::unordered_map<std::string, int> configurations;
    if (file.is_open()) {
      std::string key;
      int value; 

      while(file) {
        file >> key;
        file >> value;

        configurations[key] = value;
      }

      // print map
      // std::cout << "Using the following configurations:\n";
      // for (const auto& [key, value] : configurations)
      //   std::cout << '[' << key << "] = " << value << '\n';

      file.close();
    } else {
      std::cerr << "Error in opening file " + CONFIG_FILE + '\n';
      exit(EXIT_FAILURE);
    }

    return configurations;
  }
  // };
}

#endif // CUDACore_KernelConfigurations_h
