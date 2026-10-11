from MojoSerial.MojoBridge.DTypes import TypeableOwnedPointer

# C++ parameterises HeterogeneousSoA on a Traits policy that supplies
# `Traits::unique_ptr` and the make_unique family. The serial backend only ever
# instantiates CPUTraits, whose unique_ptr is std::unique_ptr, so the policy
# collapses to a single owning handle: TypeableOwnedPointer.
comptime HeterogeneousSoA = TypeableOwnedPointer
comptime HeterogeneousSoAImpl = TypeableOwnedPointer
comptime HeterogeneousSoACPU = HeterogeneousSoAImpl
