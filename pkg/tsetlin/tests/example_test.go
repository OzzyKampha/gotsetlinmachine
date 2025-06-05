package tsetlin_test

import (
	"fmt"
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func ExampleTsetlinMachine() {
	// Create a configuration with custom parameters
	config := &tsetlin.Config{
		NumClauses:    200, // More clauses for better accuracy
		NumFeatures:   8,   // 8 input features
		VoteThreshold: -1,  // Will be set to numClauses/2
		S:             3,   // Standard s value
		NumClasses:    2,   // Binary classification
		Epochs:        50,  // Number of training epochs
	}

	// Validate the configuration
	if err := config.Validate(); err != nil {
		fmt.Printf("Invalid configuration: %v\n", err)
		return
	}

	// Create a new Tsetlin Machine with the configuration
	tm := tsetlin.NewTsetlinMachine(
		config.NumClauses,
		config.NumFeatures,
		config.VoteThreshold,
		config.S,
	)

	// Example training data
	X := [][]int{
		{1, 0, 1, 0, 1, 0, 1, 0}, // Example input 1
		{0, 1, 0, 1, 0, 1, 0, 1}, // Example input 2
		{1, 1, 0, 0, 1, 1, 0, 0}, // Example input 3
	}
	Y := []int{1, 0, 1} // Corresponding labels

	// Train the model
	tm.Fit(X, Y, 1, config.Epochs)

	// Make a prediction
	testInput := []int{1, 0, 1, 0, 1, 0, 1, 0}
	prediction := tm.Predict(testInput)
	fmt.Printf("Prediction: %d\n", prediction)
}

func ExampleMultiClassTM() {
	// Create a configuration for multi-class classification
	config := &tsetlin.Config{
		NumClauses:    150, // Number of clauses per class
		NumFeatures:   10,  // 10 input features
		VoteThreshold: -1,  // Will be set to numClauses/2
		S:             3,   // Standard s value
		NumClasses:    3,   // Three classes
		Epochs:        100, // More epochs for multi-class
	}

	// Validate the configuration
	if err := config.Validate(); err != nil {
		fmt.Printf("Invalid configuration: %v\n", err)
		return
	}

	// Create a new Multi-Class Tsetlin Machine
	mtm := tsetlin.NewMultiClassTM(
		config.NumClasses,
		config.NumClauses,
		config.NumFeatures,
		config.VoteThreshold,
		config.S,
	)

	// Example training data
	X := [][]int{
		{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, // Class 0
		{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}, // Class 1
		{1, 1, 0, 0, 1, 1, 0, 0, 1, 1}, // Class 2
	}
	Y := []int{0, 1, 2} // Corresponding class labels

	// Train the model
	mtm.Fit(X, Y, config.Epochs)

	// Make a prediction
	testInput := []int{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}
	prediction := mtm.Predict(testInput)
	fmt.Printf("Predicted class: %d\n", prediction)
}

func TestConfigValidation(t *testing.T) {
	// Test default configuration
	config := tsetlin.DefaultConfig()
	if err := config.Validate(); err != nil {
		t.Errorf("Default configuration should be valid: %v", err)
	}

	// Test invalid configurations
	invalidConfigs := []*tsetlin.Config{
		{NumClauses: -1}, // Negative num_clauses
		{NumFeatures: 0}, // Zero num_features
		{S: -5},          // Negative s
		{NumClasses: 1},  // Less than 2 classes
		{Epochs: 0},      // Zero epochs
	}

	for i, config := range invalidConfigs {
		if err := config.Validate(); err == nil {
			t.Errorf("Configuration %d should be invalid", i)
		}
	}
}
