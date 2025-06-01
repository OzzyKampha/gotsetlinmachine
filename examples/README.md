# Tsetlin Machine Examples

This directory contains example code demonstrating how to use the Tsetlin Machine library for both binary and multiclass classification tasks.

## Running the Examples

To run the examples, use the main program:

```bash
# Run binary classification example
go run main.go binary

# Run multiclass classification example
go run main.go multiclass
```

## Binary Classification Example

The `binary/binary_classification.go` example demonstrates how to use the Tsetlin Machine for binary classification using the XOR problem. This is a classic example that shows how the Tsetlin Machine can learn non-linear patterns.

The example:
1. Creates a binary Tsetlin Machine with 2 input features
2. Trains it on the XOR dataset
3. Tests the model on the training data
4. Shows the learned clauses and their states

## Multiclass Classification Example

The `multiclass/multiclass_classification.go` example demonstrates how to use the Tsetlin Machine for multiclass classification using a simple pattern recognition problem. The example shows how to:

1. Configure a multiclass Tsetlin Machine
2. Train it on a dataset with 3 classes
3. Make predictions on both seen and unseen patterns
4. Analyze the learned clauses for each class

## Key Features Demonstrated

Both examples demonstrate:
- Configuration of the Tsetlin Machine
- Training process
- Making predictions
- Getting probability estimates
- Analyzing the learned clauses
- Error handling
- Debug logging
- Clause skipping optimization
- Parallel processing

## Performance Optimizations

The examples showcase several performance optimizations:

1. **Clause Skipping**: The implementation automatically skips clauses that don't use any of the active features in the input. This is particularly effective for sparse inputs where most features are zero.

2. **Parallel Processing**: The multiclass example demonstrates parallel training and prediction using worker pools, which can significantly improve performance on multi-core systems.

3. **Bit-level Operations**: The implementation uses bitwise operations for faster clause evaluation.

## Customizing the Examples

You can modify these examples to:
- Change the number of features
- Adjust the number of clauses and literals
- Modify the training data
- Change the classification threshold
- Adjust the specificity parameter
- Enable/disable debug logging
- Monitor clause skipping behavior
- Analyze performance with different input patterns

## Notes

- The examples use small datasets for demonstration purposes
- In real applications, you might want to:
  - Use larger datasets
  - Implement cross-validation
  - Add data preprocessing
  - Tune hyperparameters
  - Add model persistence
  - Monitor clause skipping efficiency
  - Profile performance with different input patterns

## Running Benchmarks

The library includes benchmarks to measure performance:

1. **Clause Skipping Performance**:
```bash
go test -bench=BenchmarkClauseSkipping -benchmem ./pkg/tsetlin/tests
```
This benchmark measures how clause skipping improves performance with sparse inputs.

2. **Overall Throughput**:
```bash
go test -bench=BenchmarkThroughput -benchmem ./pkg/tsetlin/tests
```
This benchmark measures the overall throughput in evaluations per second.

The benchmarks help you:
- Compare performance with and without clause skipping
- Measure the impact of different configurations
- Identify performance bottlenecks
- Optimize your specific use case

### Example Results

Recent benchmark results on an Intel Core Ultra 9 185H (5-second benchmark):

```
BenchmarkClauseSkipping-22         61021            105301 ns/op              9497 EPS         105.3 µs/eval             163 B/op          1 allocs/op
BenchmarkThroughput-22               158          36292491 ns/op            274568 EPS           0.09105 features*clauses/EPS                 3.642 µs/eval   8192295 B/op      63380 allocs/op
```

Key observations:
- Clause skipping mode:
  - ~9,497 evaluations per second
  - 105.3 microseconds per evaluation
  - Minimal memory usage (163 bytes per operation)
  - Single allocation per operation

- Full evaluation mode:
  - ~274,568 evaluations per second
  - 3.642 microseconds per evaluation
  - Higher memory usage (8MB per operation)
  - 63,380 allocations per operation

Performance trade-offs:
- Clause skipping is ~29x slower but uses ~50,000x less memory
- Full evaluation is faster but requires more memory
- Choose based on your specific requirements:
  - Use clause skipping for memory-constrained scenarios
  - Use full evaluation for maximum throughput
  - Consider hybrid approach for balanced performance 