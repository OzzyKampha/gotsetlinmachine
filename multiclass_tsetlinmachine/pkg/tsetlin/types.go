package tsetlin

import "fmt"

// PredictionResult holds the prediction results including votes for each class and the predicted class
type PredictionResult struct {
	Votes          []float64 // Votes/scores for each class
	PredictedClass int       // The predicted class
	Margin         float64   // Difference between highest and second highest votes
	Confidence     float64   // Normalized margin (margin / max_possible_votes)
}

// String returns a formatted string representation of the prediction result
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

// ClauseInfo represents information about a clause in the Tsetlin Machine
type ClauseInfo struct {
	// Literals in the clause (true for included, false for excluded)
	Literals []bool
	// State of the clause (true for positive, false for negative)
	IsPositive bool
}

// Machine represents a Tsetlin Machine classifier
type Machine interface {
	// Fit trains the model on the given data
	Fit(X [][]float64, y []int, epochs int) error

	// Predict returns the prediction results for the input
	Predict(input []float64) (PredictionResult, error)

	// PredictClass returns just the predicted class
	PredictClass(input []float64) (int, error)

	// PredictProba returns probability estimates for each class
	PredictProba(input []float64) ([]float64, error)

	// GetClauseInfo returns information about the clauses in the machine
	GetClauseInfo() [][]ClauseInfo

	// GetActiveClauses returns information about the active clauses for a given input
	GetActiveClauses(input []float64) [][]ClauseInfo
}

// Config holds the configuration parameters for a Tsetlin Machine
type Config struct {
	// Number of classes
	NumClasses int
	// Number of features
	NumFeatures int
	// Number of clauses per class
	NumClauses int
	// Number of literals per clause
	NumLiterals int
	// Threshold for classification
	Threshold float64
	// Specificity parameter
	S float64
	// Number of states for the automata
	NStates int
	// Random seed for reproducibility
	RandomSeed int64
	// Enable debug logging
	Debug bool
}

// DefaultConfig returns a Config with default values
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
