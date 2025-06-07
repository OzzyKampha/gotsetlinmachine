# Go Tsetlin Machine

Go Tsetlin Machine is a fast and thread-safe implementation of the [Tsetlin Machine](https://arxiv.org/abs/1804.01508) written in Go. It supports binary and multiclass classification with several performance optimisations for research and production use.

## Features

- Binary and multiclass learning
- Clause skipping for sparse inputs
- Bit-packed clause state representation
- Weighted voting using clause reliability (`MatchScore`) and momentum
- Safe for concurrent training and prediction
- Includes example programs and benchmarks

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
    cfg := tsetlin.DefaultConfig()
    cfg.NumFeatures = 2
    cfg.NumClauses = 10
    cfg.VoteThreshold = -1
    cfg.S = 3

    tm := tsetlin.NewTsetlinMachine(cfg.NumClauses, cfg.NumFeatures, cfg.VoteThreshold, cfg.S)

    X := [][]int{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
    y := []int{0, 1, 1, 0}
    tm.Fit(X, y, 100)

    prediction := tm.Predict([]int{0, 1})
    fmt.Println("Predicted class:", prediction)
}
```

## Examples

```bash
go run examples/binary/binary.go
go run examples/multiclass/multiclass_classification.go
```

## Development

Run tests and benchmarks using Go tools:

```bash
go test ./...
go test -bench=BenchmarkClauseSkipping -benchmem ./pkg/tsetlin/tests
```

## Project Structure

- `pkg/tsetlin` – core library
- `examples` – runnable usage examples
- `tests` – performance benchmarks

## License

MIT
