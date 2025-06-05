// Package main demonstrates multiclass classification using the Tsetlin Machine.
// This example shows how to use the Tsetlin Machine for pattern recognition
// with multiple classes, using the one-vs-all approach.
package main

import (
	"fmt"
	"math/rand"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func calculateMetrics(predictions, actual []int, numClasses int) (float64, []float64, []float64, []float64) {
	// Initialize confusion matrix
	confusion := make([][]int, numClasses)
	for i := range confusion {
		confusion[i] = make([]int, numClasses)
	}

	// Fill confusion matrix
	for i := range predictions {
		confusion[actual[i]][predictions[i]]++
	}

	// Calculate metrics for each class
	precision := make([]float64, numClasses)
	recall := make([]float64, numClasses)
	f1 := make([]float64, numClasses)

	for i := 0; i < numClasses; i++ {
		// True positives
		tp := confusion[i][i]

		// False positives (sum of column i except diagonal)
		fp := 0
		for j := 0; j < numClasses; j++ {
			if j != i {
				fp += confusion[j][i]
			}
		}

		// False negatives (sum of row i except diagonal)
		fn := 0
		for j := 0; j < numClasses; j++ {
			if j != i {
				fn += confusion[i][j]
			}
		}

		// Calculate precision
		if tp+fp > 0 {
			precision[i] = float64(tp) / float64(tp+fp)
		}

		// Calculate recall
		if tp+fn > 0 {
			recall[i] = float64(tp) / float64(tp+fn)
		}

		// Calculate F1 score
		if precision[i]+recall[i] > 0 {
			f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
		}
	}

	// Calculate macro-average F1
	macroF1 := 0.0
	for i := range f1 {
		macroF1 += f1[i]
	}
	macroF1 /= float64(numClasses)

	return macroF1, precision, recall, f1
}

func main() {
	// Training data for pattern recognition
	// Each pattern is represented by 4 binary features
	X := [][]int{
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

	// Create multiclass Tsetlin Machine
	numClasses := 3
	numClauses := 1000 // Number of clauses per class
	numFeatures := 4   // Number of input features
	threshold := 500   // Classification threshold
	s := 3             // Specificity parameter
	dropoutProb := 0.2 // Probability of dropping a clause during training

	machine := tsetlin.NewMultiClassTM(numClasses, numClauses, numFeatures, threshold, s)

	// Train the model with dropout
	fmt.Println("Training the model...")
	for epoch := 0; epoch < 1000; epoch++ {
		// Apply dropout for this epoch
		activeClauses := make([][]bool, numClasses)
		for i := range activeClauses {
			activeClauses[i] = make([]bool, numClauses)
			for j := range activeClauses[i] {
				activeClauses[i][j] = rand.Float64() > dropoutProb
			}
		}

		// Train with dropout
		for i, input := range X {
			targetClass := y[i]
			for class := 0; class < numClasses; class++ {
				// Skip dropped clauses
				if !activeClauses[class][i%numClauses] {
					continue
				}
				machine.Classes[class].Fit([][]int{input}, []int{targetClass}, class, 1)
			}
		}
	}

	// Test the model and calculate metrics
	fmt.Println("\nTesting the model...")
	predictions := make([]int, len(X))
	correct := 0
	for i, input := range X {
		predicted := machine.Predict(input)
		predictions[i] = predicted
		fmt.Printf("Input: %v, Expected: %d, Predicted: %d\n",
			input, y[i], predicted)
		if predicted == y[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(X)) * 100
	fmt.Printf("\nAccuracy: %.2f%% (%d/%d correct)\n", accuracy, correct, len(X))

	// Calculate and display detailed metrics
	macroF1, precision, recall, f1 := calculateMetrics(predictions, y, numClasses)
	fmt.Println("\nDetailed Metrics:")
	for i := 0; i < numClasses; i++ {
		fmt.Printf("Class %d:\n", i)
		fmt.Printf("  Precision: %.2f%%\n", precision[i]*100)
		fmt.Printf("  Recall: %.2f%%\n", recall[i]*100)
		fmt.Printf("  F1 Score: %.2f%%\n", f1[i]*100)
	}
	fmt.Printf("\nMacro-average F1 Score: %.2f%%\n", macroF1*100)
}
