# Patatrack Julia Serial Implementation

This is a Julia implementation of the Patatrack benchmark for CMS pixel reconstruction.

## Prerequisites

- Julia 1.11.5 or later
- Required Julia packages (installed automatically)
- Data files (downloaded automatically)

## Installation

1. **Install Julia**:
   - Download from [Julia's official website](https://julialang.org/downloads/)
   - Or use Juliaup: `curl -fsSL https://install.julialang.org | sh`
   - Set Julia 1.11.5 as default: `juliaup add 1.11.5 && juliaup default 1.11.5`

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Maya-Ali0/CMS-Julia.git
   cd CMS-Julia
   ```

3. **Setup the environment**:
   ```bash
   make julia-serial
   ```
   This will:
   - Install required Julia packages
   - Download and extract the necessary data files

## Running the code

Run the code from the `CMS-Julia` directory:
```bash
./julia-serial.sh
```

### Command Line Options

```
Usage: ./julia-serial.sh [--numberOfStreams NS] [--warmupEvents WE] [--maxEvents ME] [--runForMinutes RM]
       [--data PATH] [--validation] [--histogram] [--empty]
```

## Examples

Process 100 events with validation:
```bash
./julia-serial.sh --maxEvents 100 --validation
```

Run for 5 minutes with 8 streams:
```bash
./julia-serial.sh --runForMinutes 5 --numberOfStreams 8
```

## Troubleshooting

### Common Issues

1. **Permission denied when running ./julia-serial.sh**:
   ```bash
   chmod +x julia-serial.sh
   ./julia-serial.sh
   ```

2. **Package installation fails**:
   ```bash
   julia --project=src/julia-serial -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Precompilation gets stuck**:
   ```bash
   rm -rf ~/.julia/compiled/
   julia --project=src/julia-serial -e 'using Pkg; Pkg.precompile()'
   ```

4. **Data download fails**:
   ```bash
   make clean
   make download_raw
   ```

## Building a Standalone Application - Work in progress

To create a standalone application:
```bash
cd src/julia-serial
julia --project=. compile_app.jl
```
The executable will be created in `src/julia-serial/compile/bin/julia_main.exe`.

