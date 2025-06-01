// Package tsetlin implements the Tsetlin Machine, a novel machine learning algorithm
// that uses propositional logic to learn patterns from data.
package tsetlin

import "fmt"

// PredictionResult holds the prediction results including votes for each class and the predicted class.
// It provides a comprehensive view of the model's prediction, including confidence scores
// and voting information.
type PredictionResult struct {
	Votes          []float64 // Votes/scores for each class
	PredictedClass int       // The predicted class
	Margin         float64   // Difference between highest and second highest votes
	Confidence     float64   // Normalized margin (margin / max_possible_votes)
}

// String returns a formatted string representation of the prediction result.
// It includes the votes for each class, predicted class, margin, and confidence.
func (pr PredictionResult) String() string {
	var votesStr string
	for i, votes := range pr.Votes {
		if i > 0 {
			votesStr += ", "
		}
		votesStr += fmt.Sprintf("Class %d: %d votes", i, int(votes))
	}
	return fmt.Sprintf("Votes: [%s], Predicted Class: %d, Margin: %.2f, Confidence: %.2f",
		votesStr, pr.PredictedClass, pr.Margin, pr.Confidence)
}

// ClauseInfo represents information about a clause in the Tsetlin Machine.
// It provides details about the literals in the clause and its state.
type ClauseInfo struct {
	// Literals in the clause (true for included, false for excluded)
	Literals []bool
	// State of the clause (true for positive, false for negative)
	IsPositive bool
}

// Machine represents a Tsetlin Machine classifier interface.
// It defines the core operations that any Tsetlin Machine implementation must support.
type Machine interface {
	// Fit trains the model on the given data.
	// X is the input features matrix, y is the target labels, and epochs is the number of training iterations.
	Fit(X [][]float64, y []int, epochs int) error

	// Predict returns the prediction results for the input.
	// It includes the predicted class, confidence scores, and voting information.
	Predict(input []float64) (PredictionResult, error)

	// PredictClass returns just the predicted class for the input.
	// This is a convenience method when only the class prediction is needed.
	PredictClass(input []float64) (int, error)

	// PredictProba returns probability estimates for each class.
	// The probabilities are calculated using softmax on the voting scores.
	PredictProba(input []float64) ([]float64, error)

	// GetClauseInfo returns information about the clauses in the machine.
	// This is useful for analyzing the learned patterns and model interpretability.
	GetClauseInfo() [][]ClauseInfo

	// GetActiveClauses returns information about the active clauses for a given input.
	// This helps understand which clauses contributed to the prediction.
	GetActiveClauses(input []float64) [][]ClauseInfo
}

// Config holds the configuration parameters for a Tsetlin Machine.
// These parameters control the behavior and capacity of the model.
type Config struct {
	// Number of classes in the classification task
	NumClasses int
	// Number of input features
	NumFeatures int
	// Number of clauses per class
	NumClauses int
	// Number of literals per clause
	NumLiterals int
	// Threshold for classification
	Threshold float64
	// Specificity parameter (s) controls clause formation
	S float64
	// Number of states for the automata
	NStates int
	// Random seed for reproducibility
	RandomSeed int64
	// Enable debug logging
	Debug bool
}

// DefaultConfig returns a Config with default values.
// These defaults are suitable for many binary classification tasks.
func DefaultConfig() Config {
	return Config{
		NumClasses:  2,
		NumFeatures: 0, // Must be set by user
		NumClauses:  100,
		NumLiterals: 4,
		Threshold:   10.0,
		S:           3.9,
		NStates:     100,
		RandomSeed:  42,
		Debug:       false,
	}
}
