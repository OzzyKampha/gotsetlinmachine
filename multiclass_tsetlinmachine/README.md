# Multiclass Tsetlin Machine in Go

This project implements a multiclass Tsetlin Machine in Go. The Tsetlin Machine is a novel machine learning algorithm that uses propositional logic to learn patterns from data.

## Project Structure

```
multiclass_tsetlinmachine/
├── cmd/                    # Command-line applications
│   └── example/           # Example program demonstrating usage
├── internal/              # Private application code
│   ├── model/            # Tsetlin Machine model implementation
│   └── utils/            # Utility functions
└── pkg/                  # Public library code
```

## Features

- Multiclass classification support
- Efficient Go implementation with parallel processing
- Configurable hyperparameters:
  - Number of states (controls learning granularity)
  - Specificity parameter (s)
  - Number of clauses per class
  - Number of literals per clause
- Debug logging for detailed training insights
- Easy-to-use API

## Getting Started

### Prerequisites

- Go 1.16 or higher

### Installation

```bash
go get github.com/ozzy/multiclass_tsetlinmachine
```

### Usage

```go
package main

import (
    "github.com/ozzy/multiclass_tsetlinmachine/internal/model"
)

func main() {
    // Create a new multiclass Tsetlin Machine
    // Parameters: numClasses, numFeatures, numClauses, numLiterals, threshold, s, nStates
    mctm := model.NewMultiClassTsetlinMachine(3, 4, 10, 4, 0.5, 3.9, 20)

    // Set random seed for reproducibility
    mctm.SetRandomState(42)

    // Enable debug logging (optional)
    mctm.SetDebug(true)

    // Train the model
    mctm.Fit(X, y, epochs)

    // Make predictions
    result := mctm.Predict(input)
    fmt.Printf("Predicted Class: %d\n", result.PredictedClass)
    fmt.Printf("Confidence: %.2f\n", result.Confidence)
}
```

### Command-line Example

The example program demonstrates the usage of the Tsetlin Machine:

```bash
# Run with default settings
go run cmd/example/main.go

# Run with debug logging enabled
go run cmd/example/main.go -debug
```

### Configuration Parameters

- `numClasses`: Number of classes in the classification problem
- `numFeatures`: Number of input features
- `numClauses`: Number of clauses per class (higher values increase model capacity)
- `numLiterals`: Number of literals per clause
- `threshold`: Classification threshold
- `s`: Specificity parameter (controls clause formation)
- `nStates`: Number of states in the automaton (controls learning granularity)

### Debug Mode

When debug mode is enabled, the model provides detailed logging about:
- Active clauses and their votes
- Training progress and predictions
- Type I and Type II feedback
- State changes in the automata

This is useful for understanding the model's learning process and diagnosing issues.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 