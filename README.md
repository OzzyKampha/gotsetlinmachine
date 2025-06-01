# Go Tsetlin Machine

A Go implementation of the Tsetlin Machine, a novel machine learning algorithm that uses propositional logic to learn patterns from data. This implementation supports both binary and multiclass classification tasks.

## Project Structure

```
multiclass_tsetlinmachine/
├── pkg/                  # Public library code
│   └── tsetlin/         # Tsetlin Machine implementation
│       ├── machine.go   # Core Tsetlin Machine implementation
│       └── types.go     # Type definitions and interfaces
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

Example benchmark results on an Intel Core Ultra 9 185H (5-second benchmark):
```
BenchmarkClauseSkipping-22         61021            105301 ns/op              9497 EPS         105.3 µs/eval             163 B/op          1 allocs/op
BenchmarkThroughput-22               158          36292491 ns/op            274568 EPS           0.09105 features*clauses/EPS                 3.642 µs/eval   8192295 B/op      63380 allocs/op
```

These results show:
- Clause skipping achieves ~9,497 evaluations per second with minimal memory allocation
- Overall throughput reaches ~274,568 evaluations per second
- Memory usage is optimized for clause skipping (163 bytes/op) compared to full evaluation (8MB/op)
- The system can process ~0.09105 features*clauses per second in full evaluation mode
- Average evaluation time:
  - Clause skipping: 105.3 microseconds per evaluation
  - Full evaluation: 3.642 microseconds per evaluation

Performance characteristics:
- Clause skipping is more memory-efficient but slower per evaluation
- Full evaluation is faster per evaluation but uses more memory
- The system can handle over 270K evaluations per second in full mode
- Memory usage varies by ~50,000x between modes (163B vs 8MB per operation)

### Debugging and Analysis

- Enable debug logging by setting `config.Debug = true`
- Use `GetClauseInfo()` to analyze learned patterns
- Use `GetActiveClauses()` to understand which clauses are active for a given input
- Monitor clause skipping behavior with `CanSkipClause()`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 