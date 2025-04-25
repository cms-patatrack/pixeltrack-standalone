"""
Module for handling SiPixelGainCalibrationForHLTGPU in the Pixel detector's HLT calibration process.

# Overview
This module defines structures and functions related to gain calibration for the High-Level Trigger (HLT) on the GPU. The calibration ensures accurate signal amplification for pixel detectors, which is critical for real-time event processing in HLT systems.
"""
module CalibTrackerSiPixelESProducersInterfaceSiPixelGainCalibrationForHLTGPU


export get_cpu_product, SiPixelGainCalibrationForHLTGPU

"""
Struct to manage gain calibration data for HLT on the GPU.

This struct contains the gain calibration data and associated metadata required for the HLT processing on the GPU. It includes fields for storing the gain calibration object and the actual gain data used during the calibration process.

# Fields
- _gain_for_hlt_on_host::SiPixelGainForHLTonGPU: The gain calibration object for HLT on the host, which is used to store the gain calibration information and methods required for applying gain corrections during the HLT processing on the GPU. It ensures that the pixel data is accurately calibrated by correcting for any gain variations detected in the pixel detector.
- _gain_data::Vector{UInt8}: A vector of UInt8 representing the gain data used for calibration, which stores the actual calibration values applied to the pixel data to correct gain variations, ensuring that the detector's output is accurate and reliable.

# Constructor
Initializes the siPixelGainCalibrationForHLTGPU struct with the provided gain calibration object for HLT on the host and gain data.
"""

using ..condFormatsSiPixelObjectsSiPixelGainForHLTonGPU

struct SiPixelGainCalibrationForHLTGPU
    _gain_for_hlt_on_host::SiPixelGainForHLTonGPU # gainData is with in the SiPixelGainForHLTonGPU structure
end

"""
Getter function to access the _gain_for_hlt_on_host field of the SiPixelGainCalibrationForHLTGPU structure.
"""
get_cpu_product(calib::SiPixelGainCalibrationForHLTGPU) = calib._gain_for_hlt_on_host

end # module CalibTrackerSiPixelESProducersInterfaceSiPixelGainCalibrationForHLTGPU
