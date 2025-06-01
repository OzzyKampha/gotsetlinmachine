# Go Tsetlin Machine

A Go implementation of the Tsetlin Machine, a novel machine learning algorithm that uses propositional logic to learn patterns from data. This implementation supports both binary and multiclass classification tasks.

## Project Structure

```
pkg/                  # Public library code
└── tsetlin/         # Tsetlin Machine implementation
    ├── machine.go   # Core Tsetlin Machine implementation
    └── types.go     # Type definitions and interfaces
└── examples/            # Example code
    ├── binary/         # Binary classification example
    ├── multiclass/     # Multiclass classification example
    └── main.go         # Example runner
```

## Features

- Binary and multiclass classification support
- Thread-safe implementation
- Optimized clause skipping for improved performance
- Parallel processing with worker pools
- Configurable hyperparameters:
  - Number of states (controls learning granularity)
  - Specificity parameter (s)
  - Number of clauses
  - Number of literals per clause
  - Classification threshold
- Probability estimates for predictions
- Clause analysis and visualization
- Debug logging capabilities
- Easy-to-use API

## Installation

```bash
go get github.com/OzzyKampha/gotsetlinmachine
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func main() {
    // Create configuration
    config := tsetlin.DefaultConfig()
    config.NumFeatures = 2
    config.NumClauses = 10
    config.NumLiterals = 2
    config.Threshold = 5.0
    config.S = 3.9
    config.NStates = 100
    config.RandomSeed = 42

    // Create Tsetlin Machine
    machine, err := tsetlin.NewTsetlinMachine(config)
    if err != nil {
        log.Fatal(err)
    }

    // Train the model
    X := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    y := []int{0, 1, 1, 0}
    machine.Fit(X, y, 100)

    // Make predictions
    result, _ := machine.Predict([]float64{0, 1})
    fmt.Printf("Predicted Class: %d\n", result.PredictedClass)
    fmt.Printf("Confidence: %.2f\n", result.Confidence)
}
```

## Examples

The repository includes two example programs:

1. Binary Classification (XOR Problem):
```bash
go run examples/main.go binary
```

2. Multiclass Classification:
```bash
go run examples/main.go multiclass
```

## Configuration Parameters

- `NumFeatures`: Number of input features
- `NumClauses`: Number of clauses (higher values increase model capacity)
- `NumLiterals`: Number of literals per clause
- `Threshold`: Classification threshold
- `S`: Specificity parameter (controls clause formation)
- `NStates`: Number of states in the automaton
- `RandomSeed`: Seed for random number generation

## API Reference

### Core Functions

- `NewTsetlinMachine(config Config)`: Creates a new Tsetlin Machine
- `Fit(X [][]float64, y []int, epochs int)`: Trains the model
- `Predict(input []float64)`: Makes predictions
- `PredictClass(input []float64)`: Returns just the predicted class
- `PredictProba(input []float64)`: Returns probability estimates
- `GetClauseInfo()`: Returns information about learned clauses
- `GetActiveClauses(input []float64)`: Returns active clauses for an input
- `CanSkipClause(clauseIdx int, inputFeatureSet map[int]struct{})`: Checks if a clause can be skipped
- `InterestedFeatures(clauseIdx int)`: Returns features used by a clause

### Performance Optimizations

The implementation includes several performance optimizations:

1. **Clause Skipping**: Automatically skips clauses that don't use any of the active features in the input, significantly reducing computation time for sparse inputs.

2. **Parallel Processing**: Uses worker pools for parallel training and prediction in multiclass scenarios.

3. **Bit-level Operations**: Employs bitwise operations for faster clause evaluation.

4. **Thread Safety**: All operations are thread-safe, allowing concurrent usage.

### Benchmarking

The library includes comprehensive benchmarking tools to measure performance:

1. **Clause Skipping Benchmark**: Measures the performance impact of clause skipping with sparse inputs:
```bash
go test -bench=BenchmarkClauseSkipping -benchmem ./pkg/tsetlin/tests
```

2. **Throughput Benchmark**: Measures overall throughput in evaluations per second:
```bash
go test -bench=BenchmarkThroughput -benchmem ./pkg/tsetlin/tests
```

The benchmarks report several metrics:
- Operations per second (EPS)
- Microseconds per evaluation (µs/eval)
- Memory allocations
- Features*clauses per second

Example benchmark results on an Intel Core Ultra 9 185H:
```
BenchmarkClauseSkipping-22                174294              6726 ns/op            148681 EPS      6.726 µs/eval             0 B/op          0 allocs/op
BenchmarkLargeTsetlinMachine/Very_Sparse-22   104186             11362 ns/op      88012 EPS      11.36 µs/eval      0 B/op          0 allocs/op
BenchmarkLargeTsetlinMachine/Sparse-22        105258             11372 ns/op      87938 EPS      11.37 µs/eval      0 B/op          0 allocs/op
BenchmarkLargeTsetlinMachine/Dense-22          98154             11713 ns/op      85376 EPS      11.71 µs/eval      0 B/op          0 allocs/op
BenchmarkDCEvents-22                           49020             23913 ns/op      125469 EPS      0.02550 features*clauses/EPS   7.970 µs/event   0 B/op      0 allocs/op
BenchmarkDCEventsParallel-22                 436246              2835 ns/op      1058414 EPS     0.003023 features*clauses/EPS  0.9448 µs/event  0 B/op      0 allocs/op
BenchmarkThroughput-22                            22          49707071 ns/op      200742 EPS      0.1245 features*clauses/EPS   4.982 µs/eval    160480 B/op  10003 allocs/op
```

These results show:
- Clause skipping achieves **~148,681 evaluations per second** with zero memory allocation
- Large Tsetlin Machine benchmarks achieve **~85,000-88,000 evaluations per second**
- Memory usage is now **0 bytes/op** and **0 allocations/op** for most operations
- Average evaluation time:
  - Clause skipping: **6.726 microseconds per evaluation**
  - Large TM: **~11.4-11.7 microseconds per evaluation**
  - DC Events (Parallel): **0.9448 microseconds per event**
  - DC Events (Single): **7.970 microseconds per event**

Performance characteristics:
- Clause skipping is extremely fast and memory-efficient
- The system can process over 148,000 evaluations per second in clause skipping mode
- Parallel DC event processing achieves over 1 million events per second
- Memory usage is minimal (0 bytes/op) for most operations
- All tests pass and the code is highly efficient

### Debugging and Analysis

- Enable debug logging by setting `config.Debug = true`
- Use `GetClauseInfo()` to analyze learned patterns
- Use `GetActiveClauses()` to understand which clauses are active for a given input
- Monitor clause skipping behavior with `CanSkipClause()`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 