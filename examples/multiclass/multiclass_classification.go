package multiclass

import (
	"fmt"
	"log"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

// RunMulticlassExample demonstrates multiclass classification using pattern recognition
func RunMulticlassExample() {
	// Create configuration for multiclass classification
	config := tsetlin.DefaultConfig()
	config.NumFeatures = 4  // Number of input features
	config.NumClasses = 3   // Number of classes
	config.NumClauses = 20  // Number of clauses per class
	config.NumLiterals = 4  // Number of literals per clause
	config.Threshold = 10.0 // Classification threshold
	config.S = 3.9          // Specificity parameter
	config.NStates = 100    // Number of states for the automata
	config.RandomSeed = 42  // For reproducibility
	config.Debug = true     // Enable debug logging

	// Create multiclass Tsetlin Machine
	machine, err := tsetlin.NewMultiClassTsetlinMachine(config)
	if err != nil {
		log.Fatalf("Failed to create Multiclass Tsetlin Machine: %v", err)
	}

	// Training data for pattern recognition
	// Each pattern is represented by 4 binary features
	X := [][]float64{
		// Pattern 1: [1,1,0,0] -> Class 0
		{1, 1, 0, 0},
		{1, 1, 0, 1},
		{1, 1, 1, 0},
		// Pattern 2: [0,0,1,1] -> Class 1
		{0, 0, 1, 1},
		{1, 0, 1, 1},
		{0, 1, 1, 1},
		// Pattern 3: [1,0,1,0] -> Class 2
		{1, 0, 1, 0},
		{0, 1, 0, 1},
		{1, 0, 0, 1},
	}
	y := []int{0, 0, 0, 1, 1, 1, 2, 2, 2}

	// Train the model
	fmt.Println("Training the model...")
	if err := machine.Fit(X, y, 100); err != nil {
		log.Fatalf("Failed to train model: %v", err)
	}

	// Test the model with both training and new patterns
	testPatterns := [][]float64{
		{1, 1, 0, 0}, // Should be class 0
		{0, 0, 1, 1}, // Should be class 1
		{1, 0, 1, 0}, // Should be class 2
		{1, 1, 1, 1}, // New pattern
		{0, 0, 0, 0}, // New pattern
	}

	fmt.Println("\nTesting the model:")
	for _, input := range testPatterns {
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

		fmt.Printf("\nInput: %v\n", input)
		fmt.Printf("Predicted class: %d\n", result.PredictedClass)
		fmt.Printf("Confidence: %.2f\n", result.Confidence)
		fmt.Printf("Class probabilities: [%.3f, %.3f, %.3f]\n",
			probs[0], probs[1], probs[2])

		// Get active clauses for each class
		activeClauses := machine.GetActiveClauses(input)
		for class, clauses := range activeClauses {
			fmt.Printf("Active clauses for class %d: %d\n", class, len(clauses))
		}
	}

	// Get and print clause information for each class
	fmt.Println("\nClause Information by Class:")
	clauseInfo := machine.GetClauseInfo()
	for class, clauses := range clauseInfo {
		fmt.Printf("\nClass %d:\n", class)
		for i, clause := range clauses {
			fmt.Printf("  Clause %d:\n", i+1)
			fmt.Printf("    Is Positive: %v\n", clause.IsPositive)
			fmt.Printf("    Literals: %v\n", clause.Literals)
		}
	}
}
