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
- Configurable hyperparameters:
  - Number of states (controls learning granularity)
  - Specificity parameter (s)
  - Number of clauses
  - Number of literals per clause
  - Classification threshold
- Probability estimates for predictions
- Clause analysis and visualization
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

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 