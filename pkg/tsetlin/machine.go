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
	clauses    []*BitPackedClause
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
		clauses:            make([]*BitPackedClause, config.NumClauses),
		states:             make([][]int, config.NumClauses),
		randSource:         rand.New(rand.NewSource(config.RandomSeed)),
		interestedFeatures: make([]map[int]struct{}, config.NumClauses),
		matchScores:        make([]float64, config.NumClauses),
		momentums:          make([]float64, config.NumClauses),
	}

	// Initialize clauses and states
	for i := range tm.clauses {
		tm.clauses[i] = NewBitPackedClause(config.NumFeatures)
		tm.states[i] = make([]int, config.NumLiterals)
		tm.interestedFeatures[i] = make(map[int]struct{})
		tm.matchScores[i] = 1.0 // Initialize MatchScore to 1.0
		tm.momentums[i] = 0.5   // Initialize Momentum to 0.5

		// Ensure each clause has at least one active literal
		activeLiterals := 0
		for j := range tm.states[i] {
			// Randomly initialize literals with higher probability of being active
			if tm.randSource.Float64() < 0.7 { // 70% chance of being active
				if tm.randSource.Intn(2) == 1 {
					tm.clauses[i].SetInclude(j, true)
					activeLiterals++
					tm.interestedFeatures[i][j] = struct{}{}
				}
			}

			// Initialize states with more variation
			if tm.clauses[i].HasInclude(j) {
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
			tm.clauses[i].SetInclude(j, true)
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

				// Convert input to bit pattern
				bitInput := FromFloat64Slice(X[i])

				// Get prediction for current input
				score := 0
				for _, clause := range tm.clauses {
					if clause.Match(bitInput) {
						if clause.IsPositive {
							score++
						} else {
							score--
						}
					}
				}

				// Update states based on feedback
				predictedClass := 0
				if score < 0 {
					predictedClass = 1
				}

				// Update states for each clause
				for j, clause := range tm.clauses {
					clauseOutput := 0
					if clause.Match(bitInput) {
						clauseOutput = 1
					}

					// Update states based on feedback
					if clauseOutput == 1 {
						// Update states for matching clauses
						for k := 0; k < tm.config.NumFeatures; k++ {
							if clause.HasInclude(k) {
								tm.randMu.Lock()
								shouldUpdate := tm.randSource.Float64() < 1.0/tm.config.S
								tm.randMu.Unlock()

								if shouldUpdate {
									tm.mu.Lock()
									if y[i] == predictedClass {
										// Reward: move towards more extreme states
										if tm.states[j][k] < tm.config.NStates/2 {
											tm.states[j][k]++
										} else {
											tm.states[j][k] = tm.config.NStates
										}
									} else {
										// Penalty: move towards middle
										if tm.states[j][k] > tm.config.NStates/2 {
											tm.states[j][k]--
										} else {
											tm.states[j][k] = 1
										}
									}
									tm.mu.Unlock()
								}
							}
						}
					} else {
						// Update states for non-matching clauses
						for k := 0; k < tm.config.NumFeatures; k++ {
							if clause.HasInclude(k) {
								tm.randMu.Lock()
								shouldUpdate := tm.randSource.Float64() < 1.0/tm.config.S
								tm.randMu.Unlock()

								if shouldUpdate {
									tm.mu.Lock()
									if y[i] == predictedClass {
										// Reward: move towards middle
										if tm.states[j][k] > tm.config.NStates/2 {
											tm.states[j][k]--
										} else {
											tm.states[j][k] = 1
										}
									} else {
										// Penalty: move towards more extreme states
										if tm.states[j][k] < tm.config.NStates/2 {
											tm.states[j][k]++
										} else {
											tm.states[j][k] = tm.config.NStates
										}
									}
									tm.mu.Unlock()
								}
							}
						}
					}

					// Update match score and momentum
					tm.mu.Lock()
					if clauseOutput == 1 {
						tm.matchScores[j] = 0.9*tm.matchScores[j] + 0.1
						tm.momentums[j] = 0.9*tm.momentums[j] + 0.1
					} else {
						tm.matchScores[j] = 0.9 * tm.matchScores[j]
						tm.momentums[j] = 0.9 * tm.momentums[j]
					}
					tm.mu.Unlock()
				}
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

				// Convert input to bit pattern
				bitInput := FromFloat64Slice(X[i])

				// Calculate score
				score := 0
				for _, clause := range tm.clauses {
					if clause.Match(bitInput) {
						if clause.IsPositive {
							score++
						} else {
							score--
						}
					}
				}

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
		for i := 0; i < len(X); i++ {
			correct += <-correctChan
		}

		// Print epoch summary
		if tm.config.Debug {
			fmt.Printf("Epoch %d/%d: Accuracy = %.2f%% (%d/%d)\n",
				epoch+1, epochs, float64(correct)/float64(total)*100, correct, total)
		}
	}

	return nil
}

// Predict returns the prediction results for the input.
// It combines predictions from all clauses to make the final prediction.
func (tm *TsetlinMachine) Predict(input []float64) (PredictionResult, error) {
	if len(input) != tm.config.NumFeatures {
		return PredictionResult{}, fmt.Errorf("input features dimension mismatch: expected %d, got %d",
			tm.config.NumFeatures, len(input))
	}

	// Convert input to bit pattern
	bitInput := FromFloat64Slice(input)

	// Calculate raw score
	score := 0
	for _, clause := range tm.clauses {
		if clause.Match(bitInput) {
			if clause.IsPositive {
				score++
			} else {
				score--
			}
		}
	}

	// Determine predicted class
	predictedClass := 0
	if score < 0 {
		predictedClass = 1
	}

	// Calculate confidence using raw score and active clauses
	var activeMomentum float64
	var avgMatchScore float64
	activeClauses := 0

	// Calculate average momentum and match score for active clauses
	for i, clause := range tm.clauses {
		if clause.Match(bitInput) {
			activeMomentum += tm.momentums[i]
			avgMatchScore += tm.matchScores[i]
			activeClauses++
		}
	}

	// Calculate margin as absolute score
	margin := math.Abs(float64(score))

	// Calculate confidence based on margin and active clauses
	confidence := margin / float64(tm.config.NumClauses)
	if activeClauses > 0 {
		confidence = math.Max(confidence, float64(activeClauses)/float64(tm.config.NumClauses))
	}

	// Create votes array with raw scores, ensuring no flooring
	votes := make([]float64, 2)
	if predictedClass == 0 {
		votes[0] = float64(score)  // Keep raw score for class 0
		votes[1] = float64(-score) // Negated score for class 1
	} else {
		votes[0] = float64(-score) // Negated score for class 0
		votes[1] = float64(score)  // Keep raw score for class 1
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
			Literals:   make([]bool, tm.config.NumFeatures),
			IsPositive: clause.IsPositive,
			MatchScore: tm.matchScores[i],
			Momentum:   tm.momentums[i],
		}
		for j := 0; j < tm.config.NumFeatures; j++ {
			info[0][i].Literals[j] = clause.HasInclude(j)
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

	// Convert input to bit pattern
	bitInput := FromFloat64Slice(input)

	info := make([][]ClauseInfo, 1)
	for _, clause := range tm.clauses {
		if clause.Match(bitInput) {
			clauseInfo := ClauseInfo{
				Literals:   make([]bool, tm.config.NumFeatures),
				IsPositive: clause.IsPositive,
			}
			for j := 0; j < tm.config.NumFeatures; j++ {
				clauseInfo.Literals[j] = clause.HasInclude(j)
			}
			info[0] = append(info[0], clauseInfo)
		}
	}

	return info
}

// canSkipClause determines if a clause can be skipped during evaluation
// based on its current state and the input pattern.
func (tm *TsetlinMachine) canSkipClause(clause *BitPackedClause, input BitVec) bool {
	// Skip clause if it has no active literals
	hasActiveLiterals := false
	for i := 0; i < tm.config.NumFeatures; i++ {
		if clause.HasInclude(i) {
			hasActiveLiterals = true
			break
		}
	}
	if !hasActiveLiterals {
		return true
	}

	// Skip clause if it's a positive clause and none of its literals match
	if clause.IsPositive {
		return !clause.Match(input)
	}

	// Skip clause if it's a negative clause and all of its literals match
	return !clause.Match(input)
}

// evaluateClause evaluates a clause against the input pattern
// and returns whether the clause matches.
func (tm *TsetlinMachine) evaluateClause(clause *BitPackedClause, input BitVec) bool {
	return clause.Match(input)
}

// PrintStateInfo prints information about the current state of the machine.
func (tm *TsetlinMachine) PrintStateInfo() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	fmt.Printf("Number of clauses: %d\n", len(tm.clauses))
	fmt.Printf("Number of features: %d\n", tm.config.NumFeatures)
	fmt.Printf("Number of states: %d\n", tm.config.NStates)
	fmt.Printf("Threshold: %.2f\n", tm.config.Threshold)
	fmt.Printf("Specificity: %.2f\n", tm.config.S)

	for i, clause := range tm.clauses {
		fmt.Printf("\nClause %d:\n", i)
		fmt.Printf("  Is Positive: %v\n", clause.IsPositive)
		fmt.Printf("  Match Score: %.2f\n", tm.matchScores[i])
		fmt.Printf("  Momentum: %.2f\n", tm.momentums[i])
		fmt.Printf("  Active Literals: ")
		for j := 0; j < tm.config.NumFeatures; j++ {
			if clause.HasInclude(j) {
				fmt.Printf("%d ", j)
			}
		}
		fmt.Println()
	}
}

// printStateSummary prints a summary of the current state of the machine.
func (tm *TsetlinMachine) printStateSummary() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	fmt.Printf("Number of clauses: %d\n", len(tm.clauses))
	fmt.Printf("Number of features: %d\n", tm.config.NumFeatures)
	fmt.Printf("Number of states: %d\n", tm.config.NStates)
	fmt.Printf("Threshold: %.2f\n", tm.config.Threshold)
	fmt.Printf("Specificity: %.2f\n", tm.config.S)

	// Count active literals per clause
	activeLiterals := make([]int, len(tm.clauses))
	for i, clause := range tm.clauses {
		for j := 0; j < tm.config.NumFeatures; j++ {
			if clause.HasInclude(j) {
				activeLiterals[i]++
			}
		}
	}

	// Print summary statistics
	var totalActiveLiterals int
	for _, count := range activeLiterals {
		totalActiveLiterals += count
	}

	fmt.Printf("\nAverage active literals per clause: %.2f\n", float64(totalActiveLiterals)/float64(len(tm.clauses)))
	fmt.Printf("Total active literals: %d\n", totalActiveLiterals)
}
