# Patatrack Julia Serial Implementation

This is a Julia implementation of the Patatrack benchmark for CMS pixel reconstruction.


## Prerequisites

- Julia 1.11.5 or later  (installed automatically)
- Required Julia packages (installed automatically)
- Data files (downloaded automatically)


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cms-patatrack/pixeltrack-standalone.git
   ```

2. **Setup the environment**:
   ```bash
   make julia-serial
   ```

   This will:
   - download and install Julia 1.11.5 from [the official website](https://julialang.org/downloads/)
   - download and install required Julia packages
   - download and extract the necessary data files

## Running the code

Run the code from the top-level directory after loading the runtime environment:
```bash
source env.sh
./julia-serial
```

### Command Line Options

```
Usage: ./julia-serial [--numberOfStreams NS] [--warmupEvents WE] [--maxEvents ME] [--runForMinutes RM]
       [--data PATH] [--validation] [--histogram] [--empty]
```

For more information, run `./julia-seria --help`. 


## Examples

Process 100 events with validation:
```bash
./julia-serial --maxEvents 100 --validation
```

Run for 5 minutes with 8 streams:
```bash
./julia-serial --runForMinutes 5 --numberOfStreams 8
```


## Troubleshooting

### Common Issues

1. **julia: command not found**"
   Load the runtime environment:
   ```bash
   source env.sh
   ```

2. **Package installation fails**:
   ```bash
   julia --project=src/julia-serial -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Precompilation gets stuck**:
   ```bash
   rm -rf external/julia/depot/compiled/
   julia --project=src/julia-serial -e 'using Pkg; Pkg.precompile()'
   ```


## Building a Standalone Application - Work in progress

To create a standalone application:
```bash
cd src/julia-serial
julia --project=. compile_app.jl
```
The executable will be created in `src/julia-serial/compile/bin/julia_main.exe`.

