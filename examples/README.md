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

## Customizing the Examples

You can modify these examples to:
- Change the number of features
- Adjust the number of clauses and literals
- Modify the training data
- Change the classification threshold
- Adjust the specificity parameter
- Enable/disable debug logging

## Notes

- The examples use small datasets for demonstration purposes
- In real applications, you might want to:
  - Use larger datasets
  - Implement cross-validation
  - Add data preprocessing
  - Tune hyperparameters
  - Add model persistence 