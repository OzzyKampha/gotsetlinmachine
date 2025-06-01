package binary

import (
	"fmt"
	"log"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// RunBinaryExample demonstrates binary classification using the XOR problem
func RunBinaryExample() {
	// Create configuration for binary classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 2 // XOR has 2 inputs
	config.NumClauses = 10 // Number of clauses per class
	config.NumLiterals = 2 // Number of literals per clause (matches input features)
	config.Threshold = 5.0 // Classification threshold
	config.S = 3.9         // Specificity parameter
	config.NStates = 100   // Number of states for the automata
	config.RandomSeed = 42 // For reproducibility

	// Create binary Tsetlin Machine
	machine, err := tsetlin.NewTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Tsetlin Machine: %v", err)
	}

	// XOR training data
	X := [][]float64{
		{0, 0}, // 0
		{0, 1}, // 1
		{1, 0}, // 1
		{1, 1}, // 0
	}
	y := []int{0, 1, 1, 0}

	// Train the model
	fmt.Println("Training the model...")
	if err := machine.Fit(X, y, 100); err != nil {
		log.Fatalf("Failed to train model: %v", err)
	}

	// Test the model
	fmt.Println("\nTesting the model:")
	for i, input := range X {
		result, err := machine.Predict(input)
		if err != nil {
			log.Printf("Error predicting for input %v: %v", input, err)
			continue
		}

		probs, err := machine.PredictProba(input)
		if err != nil {
			log.Printf("Error getting probabilities for input %v: %v", input, err)
			continue
		}

		fmt.Printf("Input: %v\n", input)
		fmt.Printf("True class: %d\n", y[i])
		fmt.Printf("Predicted class: %d\n", result.PredictedClass)
		fmt.Printf("Confidence: %.2f\n", result.Confidence)
		fmt.Printf("Probabilities: [%.3f, %.3f]\n", probs[0], probs[1])
		fmt.Printf("Active clauses: %d\n", len(machine.GetActiveClauses(input)[0]))
		fmt.Println()
	}

	// Get and print clause information
	fmt.Println("Clause Information:")
	clauseInfo := machine.GetClauseInfo()
	for i, clause := range clauseInfo[0] {
		fmt.Printf("Clause %d:\n", i+1)
		fmt.Printf("  Is Positive: %v\n", clause.IsPositive)
		fmt.Printf("  Literals: %v\n", clause.Literals)
	}
}
