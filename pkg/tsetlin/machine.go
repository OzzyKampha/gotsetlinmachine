// Package tsetlin implements the Tsetlin Machine, a novel machine learning algorithm
// that uses propositional logic to learn patterns from data.
package tsetlin

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// TsetlinMachine represents a binary Tsetlin Machine.
// It implements the Machine interface and provides the core functionality
// for binary classification using Tsetlin Automata.
type TsetlinMachine struct {
	config     Config
	clauses    [][]int
	states     [][]int
	mu         sync.Mutex
	randSource *rand.Rand
	// Add interested features map for each clause
	interestedFeatures []map[int]struct{}
}

// NewTsetlinMachine creates a new binary Tsetlin Machine.
// It initializes the machine with the given configuration and sets up
// the clauses, states, and interested features tracking.
func NewTsetlinMachine(config Config) (*TsetlinMachine, error) {
	if config.NumFeatures <= 0 {
		return nil, fmt.Errorf("number of features must be positive")
	}
	if config.NumClauses <= 0 {
		return nil, fmt.Errorf("number of clauses must be positive")
	}
	if config.NumLiterals <= 0 {
		return nil, fmt.Errorf("number of literals must be positive")
	}
	if config.NStates <= 0 {
		return nil, fmt.Errorf("number of states must be positive")
	}

	tm := &TsetlinMachine{
		config:             config,
		clauses:            make([][]int, config.NumClauses),
		states:             make([][]int, config.NumClauses),
		randSource:         rand.New(rand.NewSource(config.RandomSeed)),
		interestedFeatures: make([]map[int]struct{}, config.NumClauses),
	}

	// Initialize clauses, states, and interested features
	for i := range tm.clauses {
		tm.clauses[i] = make([]int, config.NumLiterals)
		tm.states[i] = make([]int, config.NumLiterals)
		tm.interestedFeatures[i] = make(map[int]struct{})

		for j := range tm.clauses[i] {
			// Randomly initialize literals
			tm.clauses[i][j] = tm.randSource.Intn(2)
			// Initialize states to middle of range
			tm.states[i][j] = config.NStates / 2

			// Add feature to interested features if it's used in the clause
			if tm.clauses[i][j] != 0 {
				tm.interestedFeatures[i][j] = struct{}{}
			}
		}
	}

	return tm, nil
}

// Fit trains the Tsetlin Machine on the given data.
// It updates the states of the Tsetlin Automata based on the training data
// and target labels over the specified number of epochs.
func (tm *TsetlinMachine) Fit(X [][]float64, y []int, epochs int) error {
	if len(X) != len(y) {
		return fmt.Errorf("X and y must have the same length")
	}
	if len(X) == 0 {
		return fmt.Errorf("empty training data")
	}
	if len(X[0]) != tm.config.NumFeatures {
		return fmt.Errorf("input features dimension mismatch: expected %d, got %d",
			tm.config.NumFeatures, len(X[0]))
	}

	tm.mu.Lock()
	defer tm.mu.Unlock()

	for epoch := 0; epoch < epochs; epoch++ {
		for i, input := range X {
			// Get prediction for current input
			score := tm.calculateScore(input, y[i])

			// Update states based on feedback
			tm.updateStates(input, y[i], score)
		}
	}

	return nil
}

// canSkipClause checks if a clause can be skipped based on interested features.
// It uses bitwise operations for efficient feature matching.
func (tm *TsetlinMachine) canSkipClause(clauseIdx int, inputFeatures uint64) bool {
	// Use bitwise operations for faster feature matching
	var clauseFeatures uint64
	for feature := range tm.interestedFeatures[clauseIdx] {
		clauseFeatures |= 1 << feature
	}

	// If there's no overlap between clause features and input features, we can skip
	return (clauseFeatures & inputFeatures) == 0
}

// calculateScore calculates the score for a given input with feature-based clause skipping.
// It efficiently evaluates clauses using bitwise operations and clause skipping.
func (tm *TsetlinMachine) calculateScore(input []float64, target int) float64 {
	// Create input feature set using bitwise operations
	var inputFeatures uint64
	for i, val := range input {
		if val == 1 {
			inputFeatures |= 1 << i
		}
	}

	score := 0.0
	for i, clause := range tm.clauses {
		// Skip clause if none of its interested features are in the input
		if tm.canSkipClause(i, inputFeatures) {
			continue
		}

		// Evaluate clause only if it can't be skipped
		clauseOutput := tm.evaluateClause(input, clause)
		if tm.states[i][0] > tm.config.NStates/2 {
			score += float64(clauseOutput)
		} else {
			score -= float64(clauseOutput)
		}
	}
	return score
}

// evaluateClause evaluates a single clause for the given input.
// It returns 1 if the clause is satisfied, 0 otherwise.
func (tm *TsetlinMachine) evaluateClause(input []float64, clause []int) int {
	// Early exit if clause is empty
	if len(clause) == 0 {
		return 1
	}

	// Check each literal against input
	for j, literal := range clause {
		// If literal is 1 and input is 0, or literal is 0 and input is 1, clause is false
		if (literal == 1 && input[j] == 0) || (literal == 0 && input[j] == 1) {
			return 0
		}
	}

	return 1
}

// updateStates updates the states of the automata based on feedback.
// It implements Type I and Type II feedback mechanisms for learning.
func (tm *TsetlinMachine) updateStates(input []float64, target int, score float64) {
	// Type I feedback
	if (target == 1 && score < tm.config.Threshold) || (target == 0 && score > -tm.config.Threshold) {
		for i, clause := range tm.clauses {
			clauseOutput := tm.evaluateClause(input, clause)
			if clauseOutput == 1 {
				for j := range clause {
					if tm.randSource.Float64() < 1.0/tm.config.S {
						if input[j] == 1 {
							tm.states[i][j] = min(tm.states[i][j]+1, tm.config.NStates)
						} else {
							tm.states[i][j] = max(tm.states[i][j]-1, 1)
						}
					}
				}
			}
		}
	}

	// Type II feedback
	if (target == 1 && score >= tm.config.Threshold) || (target == 0 && score <= -tm.config.Threshold) {
		for i, clause := range tm.clauses {
			clauseOutput := tm.evaluateClause(input, clause)
			if clauseOutput == 1 {
				for j := range clause {
					if tm.randSource.Float64() < 1.0/tm.config.S {
						tm.states[i][j] = max(tm.states[i][j]-1, 1)
					}
				}
			}
		}
	}
}

// Predict returns the prediction results for the input.
// It includes the predicted class, confidence scores, and voting information.
func (tm *TsetlinMachine) Predict(input []float64) (PredictionResult, error) {
	if len(input) != tm.config.NumFeatures {
		return PredictionResult{}, fmt.Errorf("input features dimension mismatch: expected %d, got %d",
			tm.config.NumFeatures, len(input))
	}

	score := tm.calculateScore(input, 1)
	predictedClass := 0
	if score < 0 {
		predictedClass = 1
	}

	return PredictionResult{
		Votes:          []float64{math.Abs(score), -math.Abs(score)},
		PredictedClass: predictedClass,
		Margin:         math.Abs(score),
		Confidence:     math.Abs(score) / float64(tm.config.NumClauses),
	}, nil
}

// PredictClass returns just the predicted class for the input.
// This is a convenience method when only the class prediction is needed.
func (tm *TsetlinMachine) PredictClass(input []float64) (int, error) {
	result, err := tm.Predict(input)
	if err != nil {
		return 0, err
	}
	return result.PredictedClass, nil
}

// PredictProba returns probability estimates for each class.
// The probabilities are calculated using softmax on the voting scores.
func (tm *TsetlinMachine) PredictProba(input []float64) ([]float64, error) {
	result, err := tm.Predict(input)
	if err != nil {
		return nil, err
	}

	// Convert scores to probabilities using softmax
	expScores := make([]float64, len(result.Votes))
	var sum float64
	for i, score := range result.Votes {
		expScores[i] = math.Exp(score)
		sum += expScores[i]
	}

	probs := make([]float64, len(expScores))
	for i, expScore := range expScores {
		probs[i] = expScore / sum
	}

	return probs, nil
}

// GetClauseInfo returns information about the clauses in the machine.
// This is useful for analyzing the learned patterns and model interpretability.
func (tm *TsetlinMachine) GetClauseInfo() [][]ClauseInfo {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	info := make([][]ClauseInfo, 1)
	info[0] = make([]ClauseInfo, len(tm.clauses))

	for i, clause := range tm.clauses {
		info[0][i] = ClauseInfo{
			Literals:   make([]bool, len(clause)),
			IsPositive: tm.states[i][0] > tm.config.NStates/2,
		}
		for j, literal := range clause {
			info[0][i].Literals[j] = literal == 1
		}
	}

	return info
}

// GetActiveClauses returns information about the active clauses for a given input.
// This helps understand which clauses contributed to the prediction.
func (tm *TsetlinMachine) GetActiveClauses(input []float64) [][]ClauseInfo {
	if len(input) != tm.config.NumFeatures {
		return nil
	}

	info := make([][]ClauseInfo, 1)
	info[0] = make([]ClauseInfo, 0)

	for i, clause := range tm.clauses {
		if tm.evaluateClause(input, clause) == 1 {
			clauseInfo := ClauseInfo{
				Literals:   make([]bool, len(clause)),
				IsPositive: tm.states[i][0] > tm.config.NStates/2,
			}
			for j, literal := range clause {
				clauseInfo.Literals[j] = literal == 1
			}
			info[0] = append(info[0], clauseInfo)
		}
	}

	return info
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Clauses returns a copy of the clauses.
// This is useful for debugging and analysis.
func (tm *TsetlinMachine) Clauses() [][]int {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	clauses := make([][]int, len(tm.clauses))
	for i, clause := range tm.clauses {
		clauses[i] = make([]int, len(clause))
		copy(clauses[i], clause)
	}
	return clauses
}

// InterestedFeatures returns the interested features for a clause.
// This helps understand which features are used by each clause.
func (tm *TsetlinMachine) InterestedFeatures(clauseIdx int) map[int]struct{} {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	features := make(map[int]struct{})
	for feature := range tm.interestedFeatures[clauseIdx] {
		features[feature] = struct{}{}
	}
	return features
}

// CanSkipClause is an exported version of canSkipClause for testing.
// It checks if a clause can be skipped based on the input features.
func (tm *TsetlinMachine) CanSkipClause(clauseIdx int, inputFeatureSet map[int]struct{}) bool {
	// Convert input feature set to bits
	var inputFeatures uint64
	for feature := range inputFeatureSet {
		inputFeatures |= 1 << feature
	}
	return tm.canSkipClause(clauseIdx, inputFeatures)
}

// CalculateScore is an exported version of calculateScore for testing.
// It calculates the score for a given input with feature-based clause skipping.
func (tm *TsetlinMachine) CalculateScore(input []float64, target int) float64 {
	return tm.calculateScore(input, target)
}

// EvaluateClause is an exported version of evaluateClause for testing.
// It evaluates a single clause for the given input.
func (tm *TsetlinMachine) EvaluateClause(input []float64, clause []int) int {
	return tm.evaluateClause(input, clause)
}
