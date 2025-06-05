package tests

import (
	"fmt"
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func ExampleNewTsetlinMachine() {
	// Create a new Tsetlin Machine with 10 clauses, 5 features,
	// default vote threshold (5), and specificity parameter 3
	tm := tsetlin.NewTsetlinMachine(10, 5, -1, 3)
	fmt.Printf("Created Tsetlin Machine with %d clauses\n", len(tm.Clauses))
	// Output: Created Tsetlin Machine with 10 clauses
}

func ExampleTsetlinMachinePredict() {
	// Create a Tsetlin Machine
	tm := tsetlin.NewTsetlinMachine(10, 5, 5, 3)

	// Make a prediction
	input := []int{1, 0, 1, 0, 1}
	prediction := tm.Predict(input, 0).(int)
	fmt.Printf("Prediction: %d\n", prediction)
	// Output: Prediction: 0
}

func ExampleMultiClassTM() {
	// Create a multiclass Tsetlin Machine for 3 classes
	m := tsetlin.NewMultiClassTM(3, 10, 5, 5, 3)

	// Make a prediction
	input := []int{1, 0, 1, 0, 1}
	prediction := m.Predict(input)
	fmt.Printf("Predicted class: %d\n", prediction)
	// Output: Predicted class: 0
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
