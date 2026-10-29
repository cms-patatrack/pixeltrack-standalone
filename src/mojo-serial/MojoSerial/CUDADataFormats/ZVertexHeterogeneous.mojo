from MojoSerial.CUDADataFormats.HeterogeneousSoA import HeterogeneousSoA
from MojoSerial.CUDADataFormats.ZVertexSoA import ZVertexSoA

comptime ZVertexHeterogeneous = HeterogeneousSoA[ZVertexSoA]
