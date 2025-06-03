# Go Tsetlin Machine

A high-performance implementation of the Tsetlin Machine in Go, featuring bit-packed optimization and multiclass classification support.

## Features

- **Bit-Packed Optimization**: Efficient memory usage and faster computation through bit-level operations
- **Multiclass Classification**: Support for both binary and multiclass classification tasks
- **Thread-Safe**: Concurrent training and prediction with mutex protection
- **Configurable**: Flexible hyperparameters for fine-tuning model performance
- **Interpretable**: Built-in clause analysis for model interpretability
- **High Performance**: Optimized for speed with benchmark results:
  - Training: 800-6,700 examples/second depending on model size
  - Inference: ~2.5µs per prediction
  - Clause matching: < 1ns per operation

## Installation

```bash
go get github.com/OzzyKampha/gotsetlinmachine
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func main() {
    // Create configuration
    config := tsetlin.DefaultConfig()
    config.NumFeatures = 2  // Number of input features
    config.NumClauses = 10  // Number of clauses per class
    config.Threshold = 5.0  // Classification threshold
    config.S = 3.9         // Specificity parameter
    config.NumClasses = 2  // Binary classification

    // Create Tsetlin Machine
    machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
    if err != nil {
        log.Fatal(err)
    }

    // Training data (XOR problem)
    X := [][]float64{
        {0, 0}, // 0
        {0, 1}, // 1
        {1, 0}, // 1
        {1, 1}, // 0
    }
    y := []int{0, 1, 1, 0}

    // Train the model
    if err := machine.Fit(X, y, 100); err != nil {
        log.Fatal(err)
    }

    // Make predictions
    for _, input := range X {
        result, err := machine.Predict(input)
        if err != nil {
            log.Printf("Error predicting: %v", err)
            continue
        }
        fmt.Printf("Input: %v, Predicted: %d, Confidence: %.2f\n",
            input, result.PredictedClass, result.Confidence)
    }
}
```

## Examples

The repository includes several examples demonstrating different use cases:

1. **Binary Classification** (`examples/binary/`): XOR problem with clause analysis
2. **Multiclass Classification** (`examples/multiclass/`): Pattern recognition with multiple classes
3. **Noisy XOR** (`examples/noisyXOR/`): Learning XOR with added noise (added later)
4. **MNIST Classification** (`examples/mnist/`): Handwritten digit recognition

Each example can be run using:
```bash
go run examples/main.go [example_name]
```

## Configuration

The Tsetlin Machine can be configured using the following parameters:

- `NumFeatures`: Number of input features
- `NumClauses`: Number of clauses per class
- `NumLiterals`: Number of literals per clause
- `Threshold`: Classification threshold
- `S`: Specificity parameter
- `NStates`: Number of states for the automata
- `NumClasses`: Number of output classes
- `RandomSeed`: Random seed for reproducibility
- `Debug`: Enable debug logging

## Performance

The following benchmarks were run on an Intel Core Ultra 9 185H processor. The metrics show both throughput (operations/examples per second) and memory efficiency (bytes and allocations per operation).

### Training Speed
```
Small models:  ~850-900 examples/second
Medium models: ~4,000-4,700 examples/second
Large models:  ~6,300-6,700 examples/second
```

### Inference Performance
```
Clause matching:     1.35B ops/sec    (0 B/op, 0 allocs/op)
Full prediction:     390K ops/sec     (16 B/op, 1 allocs/op)
Clause skipping:     1.24M examples/sec (0 B/op, 0 allocs/op)
Large TM (Sparse):   1.30M examples/sec (0 B/op, 0 allocs/op)
Large TM (Dense):    911K examples/sec  (0 B/op, 0 allocs/op)
```

Key metrics explained:
- **ops/sec**: Operations per second, higher is better
- **B/op**: Bytes allocated per operation, lower is better
- **allocs/op**: Number of memory allocations per operation, lower is better
- **Sparse/Dense**: Input data density (sparse inputs are faster to process)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Tsetlin Machine algorithm by Granmo (2018)
- Inspired by the Python implementation by Ole-Christoffer Granmo 