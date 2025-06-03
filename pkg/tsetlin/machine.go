// Package tsetlin implements the Tsetlin Machine, a novel machine learning algorithm
// that uses propositional logic to learn patterns from data.
package tsetlin

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
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
	randMu     sync.Mutex // Add mutex for random number generator
	// Add interested features map for each clause
	interestedFeatures []map[int]struct{}
	// Add MatchScore and Momentum tracking for each clause
	matchScores []float64
	momentums   []float64
	// Add training lock
	isTraining bool
	// Add print mutex
	printMu sync.Mutex
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
		matchScores:        make([]float64, config.NumClauses),
		momentums:          make([]float64, config.NumClauses),
	}

	// Initialize clauses, states, and interested features
	for i := range tm.clauses {
		tm.clauses[i] = make([]int, config.NumLiterals)
		tm.states[i] = make([]int, config.NumLiterals)
		tm.interestedFeatures[i] = make(map[int]struct{})
		tm.matchScores[i] = 1.0 // Initialize MatchScore to 1.0
		tm.momentums[i] = 0.5   // Initialize Momentum to 0.5

		// Ensure each clause has at least one active literal
		activeLiterals := 0
		for j := range tm.clauses[i] {
			// Randomly initialize literals with higher probability of being active
			if tm.randSource.Float64() < 0.7 { // 70% chance of being active
				tm.clauses[i][j] = tm.randSource.Intn(2)
				if tm.clauses[i][j] != 0 {
					activeLiterals++
					tm.interestedFeatures[i][j] = struct{}{}
				}
			} else {
				tm.clauses[i][j] = 0
			}

			// Initialize states with more variation
			if tm.clauses[i][j] != 0 {
				// For active literals, initialize states more towards the extremes
				if tm.randSource.Float64() < 0.5 {
					tm.states[i][j] = tm.randSource.Intn(config.NStates/4) + 1 // Lower states
				} else {
					tm.states[i][j] = tm.randSource.Intn(config.NStates/4) + 3*config.NStates/4 // Higher states
				}
			} else {
				// For inactive literals, initialize to middle
				tm.states[i][j] = config.NStates / 2
			}
		}

		// If no active literals, force at least one
		if activeLiterals == 0 {
			j := tm.randSource.Intn(config.NumLiterals)
			tm.clauses[i][j] = 1
			tm.interestedFeatures[i][j] = struct{}{}
			tm.states[i][j] = tm.randSource.Intn(config.NStates/4) + 3*config.NStates/4
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
	if tm.isTraining {
		tm.mu.Unlock()
		return fmt.Errorf("training already in progress")
	}
	tm.isTraining = true
	tm.mu.Unlock()

	defer func() {
		tm.mu.Lock()
		tm.isTraining = false
		tm.mu.Unlock()
	}()

	// Print initial state summary
	if tm.config.Debug {
		fmt.Printf("\nInitial state summary:\n")
		tm.printStateSummary()
	}

	for epoch := 0; epoch < epochs; epoch++ {
		correct := 0
		total := len(X)

		// Create worker pool for parallel processing
		numWorkers := runtime.NumCPU() * 2
		workerChan := make(chan int, numWorkers)
		for i := 0; i < numWorkers; i++ {
			workerChan <- i
		}

		// First pass: update states in parallel
		var wg sync.WaitGroup
		for i := 0; i < len(X); i++ {
			i := i // Create new variable for goroutine
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-workerChan
				defer func() { workerChan <- 0 }()

				// Get prediction for current input
				score := tm.calculateScore(X[i], y[i])
				// Update states based on feedback
				tm.updateStates(X[i], y[i], score)
			}()
		}
		wg.Wait()

		// Second pass: calculate accuracy in parallel
		correctChan := make(chan int, numWorkers)
		for i := 0; i < len(X); i++ {
			i := i // Create new variable for goroutine
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-workerChan
				defer func() { workerChan <- 0 }()

				score := tm.calculateScore(X[i], y[i])
				predictedClass := 0
				if score < 0 {
					predictedClass = 1
				}
				if predictedClass == y[i] {
					correctChan <- 1
				} else {
					correctChan <- 0
				}
			}()
		}

		// Collect results
		go func() {
			wg.Wait()
			close(correctChan)
		}()

		// Count correct predictions
		for c := range correctChan {
			correct += c
		}

		// Calculate and print accuracy for this epoch with print mutex
		accuracy := float64(correct) / float64(total)
		tm.printMu.Lock()
		fmt.Printf("Epoch %d/%d - Accuracy: %.2f%% (%d/%d)\n",
			epoch+1, epochs, accuracy*100, correct, total)
		tm.printMu.Unlock()

		// Print state summary after each epoch if debug is enabled
		if tm.config.Debug && (epoch == 0 || epoch == epochs-1) {
			fmt.Printf("\nState summary after epoch %d:\n", epoch+1)
			tm.printStateSummary()
		}
	}

	return nil
}

// printStateSummary prints a summary of the current state distribution
func (tm *TsetlinMachine) printStateSummary() {
	// Calculate sum of states
	var totalSum int
	histogram := make([]int, 5) // 5 bins: 0-20, 21-40, 41-60, 61-80, 81-100

	for _, clauseStates := range tm.states {
		for _, state := range clauseStates {
			totalSum += state
			// Add to appropriate histogram bin
			bin := state / 20
			if bin > 4 {
				bin = 4
			}
			histogram[bin]++
		}
	}

	// Print summary
	fmt.Printf("Total states sum: %d\n", totalSum)
	fmt.Printf("State distribution:\n")
	fmt.Printf("  0-20:  %d states\n", histogram[0])
	fmt.Printf(" 21-40:  %d states\n", histogram[1])
	fmt.Printf(" 41-60:  %d states\n", histogram[2])
	fmt.Printf(" 61-80:  %d states\n", histogram[3])
	fmt.Printf(" 81-100: %d states\n", histogram[4])
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
	activeClauses := 0

	for i, clause := range tm.clauses {
		// Skip clause if none of its interested features are in the input
		if tm.canSkipClause(i, inputFeatures) {
			// Update MatchScore and Momentum for skipped clauses
			tm.matchScores[i] *= 0.9 // Decay MatchScore
			tm.momentums[i] *= 0.8   // Faster decay for inactive clauses
			continue
		}

		// Evaluate clause only if it can't be skipped
		clauseOutput := tm.evaluateClause(input, clause)

		if clauseOutput == 1 {
			// Update MatchScore and Momentum for matched clauses
			tm.matchScores[i] = 0.9*tm.matchScores[i] + 0.1 // Decay + small increment
			tm.momentums[i] = 0.8*tm.momentums[i] + 0.2     // Faster decay + increment

			// Add raw score based on clause state
			if tm.states[i][0] > tm.config.NStates/2 {
				score += 1.0
			} else {
				score -= 1.0
			}
			activeClauses++
		} else {
			// Update MatchScore and Momentum for unmatched clauses
			tm.matchScores[i] *= 0.9 // Decay MatchScore
			tm.momentums[i] *= 0.8   // Faster decay for inactive clauses
		}
	}

	return score
}

// calculateScoreReadOnly calculates the score for a given input with feature-based clause skipping, without mutating any state.
func (tm *TsetlinMachine) calculateScoreReadOnly(input []float64) float64 {
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
		// Use current MatchScore for weighting, but do not update it
		// Normalize MatchScore to be between 0 and 1 for weighting
		normalizedScore := tm.matchScores[i] / (tm.matchScores[i] + 1.0) // Sigmoid-like normalization
		if tm.states[i][0] > tm.config.NStates/2 {
			score += float64(clauseOutput) * normalizedScore
		} else {
			score -= float64(clauseOutput) * normalizedScore
		}
	}
	return score
}

// evaluateClause evaluates a single clause for the given input.
// It returns 1 if the clause is satisfied, 0 otherwise.
func (tm *TsetlinMachine) evaluateClause(input []float64, clause []int) int {
	// Count active literals
	activeLiterals := 0
	matches := 0

	// Check each literal against input
	for j, literal := range clause {
		if literal != 0 { // Count non-zero literals as active
			activeLiterals++
			// If literal is 1 and input is 1, or literal is 0 and input is 0, it's a match
			if (literal == 1 && input[j] == 1) || (literal == 0 && input[j] == 0) {
				matches++
			}
		}
	}

	// Require at least one active literal
	if activeLiterals == 0 {
		return 0
	}

	// Calculate match ratio
	matchRatio := float64(matches) / float64(activeLiterals)

	// Return 1 if match ratio is above threshold (e.g., 0.7)
	if matchRatio >= 0.7 {
		return 1
	}
	return 0
}

// updateStates updates the states of the automata based on feedback.
// It implements Type I and Type II feedback mechanisms for learning.
func (tm *TsetlinMachine) updateStates(input []float64, target int, score float64) {
	updates := 0

	// Type I feedback - when prediction is incorrect
	if (target == 1 && score < tm.config.Threshold) || (target == 0 && score > -tm.config.Threshold) {
		for i, clause := range tm.clauses {
			clauseOutput := tm.evaluateClause(input, clause)

			// Update states for both matching and non-matching clauses
			for j := range clause {
				tm.randMu.Lock()
				shouldUpdate := tm.randSource.Float64() < 1.0/tm.config.S
				tm.randMu.Unlock()

				if shouldUpdate {
					if clauseOutput == 1 {
						// For matching clauses
						if target == 1 {
							// Positive feedback for positive examples
							old := tm.states[i][j]
							// More conservative state updates
							if tm.states[i][j] < tm.config.NStates/2 {
								tm.states[i][j] = min(tm.states[i][j]+1, tm.config.NStates/2)
							} else {
								tm.states[i][j] = min(tm.states[i][j]+1, tm.config.NStates)
							}
							if tm.states[i][j] != old {
								updates++
							}
						} else {
							// Negative feedback for negative examples
							old := tm.states[i][j]
							// More conservative state updates
							if tm.states[i][j] > tm.config.NStates/2 {
								tm.states[i][j] = max(tm.states[i][j]-1, tm.config.NStates/2)
							} else {
								tm.states[i][j] = max(tm.states[i][j]-1, 1)
							}
							if tm.states[i][j] != old {
								updates++
							}
						}
					} else {
						// For non-matching clauses
						if target == 0 {
							// Positive feedback for negative examples
							old := tm.states[i][j]
							// More conservative state updates
							if tm.states[i][j] < tm.config.NStates/2 {
								tm.states[i][j] = min(tm.states[i][j]+1, tm.config.NStates/2)
							} else {
								tm.states[i][j] = min(tm.states[i][j]+1, tm.config.NStates)
							}
							if tm.states[i][j] != old {
								updates++
							}
						} else {
							// Negative feedback for positive examples
							old := tm.states[i][j]
							// More conservative state updates
							if tm.states[i][j] > tm.config.NStates/2 {
								tm.states[i][j] = max(tm.states[i][j]-1, tm.config.NStates/2)
							} else {
								tm.states[i][j] = max(tm.states[i][j]-1, 1)
							}
							if tm.states[i][j] != old {
								updates++
							}
						}
					}
				}
			}

			// Update match score with decay
			if clauseOutput == 1 {
				tm.matchScores[i] = 0.95*tm.matchScores[i] + 0.05 // Slower decay + smaller increment
				tm.momentums[i] = 0.9*tm.momentums[i] + 0.1       // Slower decay + smaller increment
			} else {
				tm.matchScores[i] *= 0.95 // Slower decay for non-matching
				tm.momentums[i] *= 0.9    // Slower decay for non-matching
			}
		}
	}

	// Type II feedback - when prediction is correct
	if (target == 1 && score >= tm.config.Threshold) || (target == 0 && score <= -tm.config.Threshold) {
		for i, clause := range tm.clauses {
			clauseOutput := tm.evaluateClause(input, clause)
			if clauseOutput == 1 {
				for j := range clause {
					tm.randMu.Lock()
					shouldUpdate := tm.randSource.Float64() < 1.0/tm.config.S
					tm.randMu.Unlock()

					if shouldUpdate {
						old := tm.states[i][j]
						// More conservative state updates for Type II feedback
						if tm.states[i][j] > tm.config.NStates/2 {
							tm.states[i][j] = max(tm.states[i][j]-1, tm.config.NStates/2)
						} else {
							tm.states[i][j] = max(tm.states[i][j]-1, 1)
						}
						if tm.states[i][j] != old {
							updates++
						}
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

	// Calculate raw score
	score := tm.calculateScore(input, 1)

	// Determine predicted class
	predictedClass := 0
	if score < 0 {
		predictedClass = 1
	}

	// Calculate confidence using raw score and active clauses
	var activeMomentum float64
	var avgMatchScore float64
	activeClauses := 0

	// Create input feature set using bitwise operations
	var inputFeatures uint64
	for i, val := range input {
		if val == 1 {
			inputFeatures |= 1 << i
		}
	}

	// Calculate average momentum and match score for active clauses
	for i := range tm.clauses {
		if !tm.canSkipClause(i, inputFeatures) {
			clauseOutput := tm.evaluateClause(input, tm.clauses[i])
			if clauseOutput == 1 {
				activeMomentum += tm.momentums[i]
				avgMatchScore += tm.matchScores[i]
				activeClauses++
			}
		}
	}

	// Calculate margin as absolute score
	margin := math.Abs(score)

	// Calculate confidence based on margin and active clauses
	confidence := margin / float64(tm.config.NumClauses)
	if activeClauses > 0 {
		confidence = math.Max(confidence, float64(activeClauses)/float64(tm.config.NumClauses))
	}

	// Create votes array with raw scores, ensuring no flooring
	votes := make([]float64, 2)
	if predictedClass == 0 {
		votes[0] = score  // Keep raw score for class 0
		votes[1] = -score // Negated score for class 1
	} else {
		votes[0] = -score // Negated score for class 0
		votes[1] = score  // Keep raw score for class 1
	}

	// Ensure votes are not zero if there are active clauses
	if activeClauses > 0 && math.Abs(votes[0]) < 1e-10 && math.Abs(votes[1]) < 1e-10 {
		// If votes are too small, scale them up based on active clauses
		scale := float64(activeClauses) / float64(tm.config.NumClauses)
		if predictedClass == 0 {
			votes[0] = scale
			votes[1] = -scale
		} else {
			votes[0] = -scale
			votes[1] = scale
		}
	}

	return PredictionResult{
		Votes:          votes,
		PredictedClass: predictedClass,
		Margin:         margin,
		Confidence:     confidence,
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
			MatchScore: tm.matchScores[i],
			Momentum:   tm.momentums[i],
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

// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the larger of two integers
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

// PrintStateInfo prints information about the current state of the machine
func (tm *TsetlinMachine) PrintStateInfo() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	var totalSum int
	histogram := make([]int, 5) // 5 bins: 0-20, 21-40, 41-60, 61-80, 81-100

	for _, clauseStates := range tm.states {
		for _, state := range clauseStates {
			totalSum += state
			bin := state / 20
			if bin > 4 {
				bin = 4
			}
			histogram[bin]++
		}
	}

	fmt.Printf("Total states sum: %d\n", totalSum)
	fmt.Printf("State distribution:\n")
	fmt.Printf("  0-20:  %d states\n", histogram[0])
	fmt.Printf(" 21-40:  %d states\n", histogram[1])
	fmt.Printf(" 41-60:  %d states\n", histogram[2])
	fmt.Printf(" 61-80:  %d states\n", histogram[3])
	fmt.Printf(" 81-100: %d states\n", histogram[4])
}
