package tsetlin

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// TsetlinMachine represents a binary Tsetlin Machine
type TsetlinMachine struct {
	config     Config
	clauses    [][]int
	states     [][]int
	mu         sync.Mutex
	randSource *rand.Rand
}

// NewTsetlinMachine creates a new binary Tsetlin Machine
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
		config:     config,
		clauses:    make([][]int, config.NumClauses),
		states:     make([][]int, config.NumClauses),
		randSource: rand.New(rand.NewSource(config.RandomSeed)),
	}

	// Initialize clauses and states
	for i := range tm.clauses {
		tm.clauses[i] = make([]int, config.NumLiterals)
		tm.states[i] = make([]int, config.NumLiterals)
		for j := range tm.clauses[i] {
			// Randomly initialize literals
			tm.clauses[i][j] = tm.randSource.Intn(2)
			// Initialize states to middle of range
			tm.states[i][j] = config.NStates / 2
		}
	}

	return tm, nil
}

// Fit trains the Tsetlin Machine on the given data
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

// calculateScore calculates the score for a given input
func (tm *TsetlinMachine) calculateScore(input []float64, target int) float64 {
	score := 0.0
	for i, clause := range tm.clauses {
		clauseOutput := tm.evaluateClause(input, clause)
		if tm.states[i][0] > tm.config.NStates/2 {
			score += float64(clauseOutput)
		} else {
			score -= float64(clauseOutput)
		}
	}
	return score
}

// evaluateClause evaluates a single clause for the given input
func (tm *TsetlinMachine) evaluateClause(input []float64, clause []int) int {
	result := 1
	for j, literal := range clause {
		if literal == 1 {
			if input[j] == 0 {
				result = 0
				break
			}
		} else {
			if input[j] == 1 {
				result = 0
				break
			}
		}
	}
	return result
}

// updateStates updates the states of the automata based on feedback
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

// Predict returns the prediction results for the input
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

// PredictClass returns just the predicted class
func (tm *TsetlinMachine) PredictClass(input []float64) (int, error) {
	result, err := tm.Predict(input)
	if err != nil {
		return 0, err
	}
	return result.PredictedClass, nil
}

// PredictProba returns probability estimates for each class
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

// GetClauseInfo returns information about the clauses in the machine
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

// GetActiveClauses returns information about the active clauses for a given input
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
