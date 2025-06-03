// Package tsetlin implements the Tsetlin Machine, a novel machine learning algorithm
// that uses propositional logic to learn patterns from data.
//
// The Tsetlin Machine is a powerful machine learning algorithm that combines
// the interpretability of rule-based systems with the learning capabilities
// of neural networks. It uses teams of Tsetlin Automata to learn patterns
// in data and make predictions.
//
// Key Features:
//   - Binary and multiclass classification support
//   - Thread-safe implementation
//   - Optimized clause skipping for improved performance
//   - Parallel processing with worker pools
//   - Configurable hyperparameters
//   - Probability estimates for predictions
//   - Clause analysis and visualization
//
// Example usage:
//
//	config := tsetlin.DefaultConfig()
//	config.NumFeatures = 2
//	config.NumClauses = 10
//	config.NumLiterals = 2
//	config.Threshold = 5.0
//	config.S = 3.9
//	config.NStates = 100
//	config.NumClasses = 2  // Binary classification
//
//	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Train the model
//	X := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
//	y := []int{0, 1, 1, 0}
//	machine.Fit(X, y, 100)
//
//	// Make predictions
//	result, _ := machine.Predict([]float64{0, 1})
//	fmt.Printf("Predicted Class: %d\n", result.PredictedClass)
//	fmt.Printf("Confidence: %.2f\n", result.Confidence)
//
// For more examples and detailed documentation, visit:
// https://github.com/OzzyKampha/gotsetlinmachine
package tsetlin
