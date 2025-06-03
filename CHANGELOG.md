# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive benchmark suite for performance testing
- Detailed performance metrics for training and inference
- Clause analysis functionality for model interpretability
- Confidence scores in prediction results
- Bit-packed optimization for improved performance
- New noisy XOR example with training and test datasets
- Parallel processing with worker pools for training

### Changed
- Unified constructor to use `NewMultiClassTsetlinMachine` as main entry point
- Updated all examples to use the new constructor
- Improved documentation and code comments
- Enhanced error handling in prediction methods
- Optimized clause skipping for better performance
- Refactored machine.go for better code organization
- Removed sharded inference in favor of bit-packed optimization

### Performance
- Training speed:
  - Small models: ~850-900 examples/second
  - Medium models: ~4,000-4,700 examples/second
  - Large models: ~6,300-6,700 examples/second
- Inference speed:
  - Basic clause matching: 0.69 ns/op
  - Full prediction: 2,546 ns/op
  - Clause skipping optimization: 780.1 ns/op

### Fixed
- Constructor type mismatch in examples
- Prediction return value handling in tests
- Documentation consistency across package
- Memory efficiency with bit-packed optimization

## [0.1.0] - 2024-03-19

### Added
- Initial release of Go Tsetlin Machine implementation
- Basic binary and multiclass classification support
- Bit-packed optimization for improved performance
- Thread-safe implementation with mutex protection
- Configurable hyperparameters
- Example implementations for XOR, MNIST, and pattern recognition
- Comprehensive test suite
- Basic documentation and examples

### Performance
- Initial benchmark results:
  - Clause matching: < 1ns per operation
  - Full prediction: ~2.5µs per operation
  - Training: 800-6,700 examples per second depending on model size

### Changed
- None

### Deprecated
- None

### Removed
- None

### Security
- None 