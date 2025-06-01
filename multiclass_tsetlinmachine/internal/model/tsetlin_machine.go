package model

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"time"
)

// TsetlinAutomaton represents the state automaton for each literal
type TsetlinAutomaton struct {
	state   int
	nStates int
}

// reward moves the automaton towards the include state
func (ta *TsetlinAutomaton) reward() {
	ta.state = min(ta.state+1, ta.nStates)
}

// penalty moves the automaton towards the exclude state
func (ta *TsetlinAutomaton) penalty() {
	ta.state = max(ta.state-1, 0)
}

// getAction returns 1 for include, -1 for exclude
func (ta *TsetlinAutomaton) getAction() int {
	if ta.state > ta.nStates/2 {
		return 1
	}
	return -1
}

// WorkerPool represents a pool of worker goroutines
type WorkerPool struct {
	numWorkers int
	tasks      chan func()
	wg         sync.WaitGroup
	done       chan struct{}
}

// NewWorkerPool creates a new worker pool with the specified number of workers
func NewWorkerPool(numWorkers int) *WorkerPool {
	pool := &WorkerPool{
		numWorkers: numWorkers,
		tasks:      make(chan func()),
		done:       make(chan struct{}),
	}
	pool.start()
	return pool
}

// start initializes the worker goroutines
func (wp *WorkerPool) start() {
	for i := 0; i < wp.numWorkers; i++ {
		go func() {
			for {
				select {
				case task, ok := <-wp.tasks:
					if !ok {
						return
					}
					task()
					wp.wg.Done()
				case <-wp.done:
					return
				}
			}
		}()
	}
}

// Submit adds a task to the worker pool
func (wp *WorkerPool) Submit(task func()) {
	wp.wg.Add(1)
	wp.tasks <- task
}

// Wait blocks until all submitted tasks are completed
func (wp *WorkerPool) Wait() {
	wp.wg.Wait()
}

// Close shuts down the worker pool
func (wp *WorkerPool) Close() {
	close(wp.done)
	close(wp.tasks)
}

// Global worker pool for all Tsetlin Machine instances
var globalWorkerPool *WorkerPool

// InitGlobalPool initializes the global worker pool with the specified number of workers
func InitGlobalPool(numWorkers int) {
	if globalWorkerPool != nil {
		globalWorkerPool.Close()
	}
	globalWorkerPool = NewWorkerPool(numWorkers)
}

// CloseGlobalPool closes the global worker pool
func CloseGlobalPool() {
	if globalWorkerPool != nil {
		globalWorkerPool.Close()
		globalWorkerPool = nil
	}
}

// TsetlinMachine represents a multiclass Tsetlin Machine
type TsetlinMachine struct {
	// Number of classes
	numClasses int
	// Number of features
	numFeatures int
	// Number of clauses per class
	numClauses int
	// Number of literals per clause
	numLiterals int
	// Threshold for classification
	threshold float64
	// Specificity parameter (s)
	s float64
	// Number of states in the automaton
	nStates int
	// Clauses for each class
	clauses [][]int
	// State automata for each literal
	automata [][]TsetlinAutomaton
	// Clause signs (+1 or -1) for each class
	clauseSigns [][]int
	// Random number generator
	rng *rand.Rand
	// Worker pool
	numWorkers int
	// Mutex for thread-safe operations
	mu sync.Mutex
	// Debug flag for logging
	debug bool
	// Bitfields for clause skipping optimization
	clauseBitSets [][]uint64
	// Flag to indicate if we're in training mode
	isTraining bool
}

// NewTsetlinMachine creates a new Tsetlin Machine instance
func NewTsetlinMachine(numClasses, numFeatures, numClauses, numLiterals int, threshold, s float64) *TsetlinMachine {
	tm := &TsetlinMachine{
		numClasses:  numClasses,
		numFeatures: numFeatures,
		numClauses:  numClauses,
		numLiterals: numLiterals,
		threshold:   threshold,
		s:           s,
		nStates:     100, // Default number of states
		rng:         rand.New(rand.NewSource(42)),
		numWorkers:  runtime.NumCPU(), // Use number of CPU cores
		debug:       false,            // Debug logging off by default
	}

	// Initialize clauses, automata, and signs
	tm.initialize()

	return tm
}

// SetRandomState sets the random seed for reproducibility
func (tm *TsetlinMachine) SetRandomState(seed int64) {
	tm.rng = rand.New(rand.NewSource(seed))
}

// SetNStates sets the number of states in the automaton
func (tm *TsetlinMachine) SetNStates(nStates int) {
	tm.nStates = nStates
	tm.initialize() // Reinitialize with new number of states
}

// SetS sets the specificity parameter
func (tm *TsetlinMachine) SetS(s float64) {
	tm.s = s
}

// SetDebug sets the debug flag for logging
func (tm *TsetlinMachine) SetDebug(debug bool) {
	tm.debug = debug
}

// initialize sets up the initial state of the Tsetlin Machine
func (tm *TsetlinMachine) initialize() {
	// Initialize clauses for each class
	tm.clauses = make([][]int, tm.numClasses)
	for i := range tm.clauses {
		tm.clauses[i] = make([]int, tm.numClauses)
	}

	// Initialize automata
	tm.automata = make([][]TsetlinAutomaton, tm.numClasses)
	for i := range tm.automata {
		tm.automata[i] = make([]TsetlinAutomaton, tm.numClauses*tm.numLiterals)
		for j := range tm.automata[i] {
			tm.automata[i][j] = TsetlinAutomaton{
				state:   tm.nStates / 2, // Start in middle state
				nStates: tm.nStates,
			}
		}
	}

	// Initialize clause signs
	tm.clauseSigns = make([][]int, tm.numClasses)
	for i := range tm.clauseSigns {
		tm.clauseSigns[i] = make([]int, tm.numClauses)
		// Randomly assign signs (+1 or -1) to clauses
		for j := range tm.clauseSigns[i] {
			if tm.rng.Float64() < 0.5 {
				tm.clauseSigns[i][j] = 1
			} else {
				tm.clauseSigns[i][j] = -1
			}
		}
	}

	// Initialize bitfields for clause skipping
	tm.initializeBitSets()
}

// initializeBitSets initializes the bitfields for clause skipping
func (tm *TsetlinMachine) initializeBitSets() {
	// Calculate number of uint64s needed (64 bits per uint64)
	numBits := (tm.numFeatures + 63) / 64
	tm.clauseBitSets = make([][]uint64, tm.numClasses)

	for class := 0; class < tm.numClasses; class++ {
		tm.clauseBitSets[class] = make([]uint64, tm.numClauses*numBits)

		// Initialize bitfields for each clause
		for clause := 0; clause < tm.numClauses; clause++ {
			for literal := 0; literal < tm.numLiterals; literal++ {
				idx := clause*tm.numLiterals + literal
				action := tm.automata[class][idx].getAction()

				if action == 1 {
					// Set bit for positive literal
					bitPos := literal
					uint64Idx := bitPos / 64
					bitOffset := bitPos % 64
					tm.clauseBitSets[class][clause*numBits+uint64Idx] |= 1 << bitOffset
				}
			}
		}
	}
}

// updateBitSets updates the bitfields after training
func (tm *TsetlinMachine) updateBitSets(class, clause int) {
	numBits := (tm.numFeatures + 63) / 64
	uint64Idx := clause * numBits

	// Clear existing bits
	for i := 0; i < numBits; i++ {
		tm.clauseBitSets[class][uint64Idx+i] = 0
	}

	// Set new bits based on current automaton states
	for literal := 0; literal < tm.numLiterals; literal++ {
		idx := clause*tm.numLiterals + literal
		action := tm.automata[class][idx].getAction()

		if action == 1 {
			bitPos := literal
			uint64Idx := bitPos / 64
			bitOffset := bitPos % 64
			tm.clauseBitSets[class][clause*numBits+uint64Idx] |= 1 << bitOffset
		}
	}
}

// canSkipClause checks if a clause can be skipped for a given input
func (tm *TsetlinMachine) canSkipClause(input []float64, class, clause int) bool {
	numBits := (tm.numFeatures + 63) / 64
	uint64Idx := clause * numBits

	// Convert input to bitfield
	var inputBits []uint64
	for i := 0; i < numBits; i++ {
		var bits uint64
		for j := 0; j < 64 && i*64+j < len(input); j++ {
			if input[i*64+j] > 0.5 {
				bits |= 1 << j
			}
		}
		inputBits = append(inputBits, bits)
	}

	// Check if there are ANY matching features
	hasMatch := false
	for i := 0; i < numBits; i++ {
		requiredBits := tm.clauseBitSets[class][uint64Idx+i]
		actualBits := inputBits[i]

		// If there's any overlap between required and actual bits, we have a match
		if (actualBits & requiredBits) != 0 {
			hasMatch = true
			break
		}
	}

	// Skip if there are NO matches
	if !hasMatch {
		if tm.debug {
			// Show which features were required
			var requiredFeatures []int
			for i := 0; i < numBits; i++ {
				requiredBits := tm.clauseBitSets[class][uint64Idx+i]
				for j := 0; j < 64 && i*64+j < tm.numFeatures; j++ {
					if requiredBits&(1<<j) != 0 {
						requiredFeatures = append(requiredFeatures, i*64+j)
					}
				}
			}
			log.Printf("Clause %d (class %d) skipped: No matching features. Required features: %v",
				clause, class, requiredFeatures)
		}
		return true
	}

	return false
}

// calculateScore computes the number of votes for a given class and input
func (tm *TsetlinMachine) calculateScore(input []float64, class int) float64 {
	// Pre-compute input bits once and reuse the slice
	numBits := (tm.numFeatures + 63) / 64
	inputBits := make([]uint64, numBits)

	// Use bit operations for faster conversion
	for i := 0; i < numBits; i++ {
		var bits uint64
		end := min(64, tm.numFeatures-i*64)
		for j := 0; j < end; j++ {
			if input[i*64+j] > 0.5 {
				bits |= 1 << j
			}
		}
		inputBits[i] = bits
	}

	// Process clauses in larger chunks to reduce task scheduling overhead
	chunkSize := 100 // Increased chunk size for better throughput
	numChunks := (tm.numClauses + chunkSize - 1) / chunkSize
	votes := make([]float64, numChunks)
	var wg sync.WaitGroup
	wg.Add(numChunks)

	// Create a pool of workers for processing chunks
	workerChan := make(chan int, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		workerChan <- i
	}

	// Pre-allocate clause evaluation result
	clauseActive := true

	for chunk := 0; chunk < numChunks; chunk++ {
		chunk := chunk // Create new variable for goroutine
		<-workerChan   // Get worker from pool
		go func() {
			defer wg.Done()
			defer func() { workerChan <- 0 }() // Return worker to pool

			startClause := chunk * chunkSize
			endClause := min((chunk+1)*chunkSize, tm.numClauses)
			localVotes := 0.0

			for clause := startClause; clause < endClause; clause++ {
				// Skip clause if it can't match (only during inference)
				if !tm.isTraining {
					uint64Idx := clause * numBits
					hasMatch := false

					// Use SIMD-like operations for bit matching
					for i := 0; i < numBits; i++ {
						if (inputBits[i] & tm.clauseBitSets[class][uint64Idx+i]) != 0 {
							hasMatch = true
							break
						}
					}
					if !hasMatch {
						continue
					}
				}

				clauseActive = true
				// Unroll the literal loop for better performance
				for literal := 0; literal < tm.numLiterals; literal += 4 {
					// Process 4 literals at a time
					for j := 0; j < 4 && literal+j < tm.numLiterals; j++ {
						idx := clause*tm.numLiterals + literal + j
						action := tm.automata[class][idx].getAction()

						if action == 1 {
							if input[literal+j] == 0 {
								clauseActive = false
								goto ClauseDone
							}
						} else {
							if input[literal+j] == 1 {
								clauseActive = false
								goto ClauseDone
							}
						}
					}
				}
			ClauseDone:
				if clauseActive {
					localVotes += float64(tm.clauseSigns[class][clause])
				}
			}
			votes[chunk] = localVotes
		}()
	}

	// Wait for all chunks to complete
	wg.Wait()

	// Sum up votes from all chunks
	totalVotes := 0.0
	for _, v := range votes {
		totalVotes += v
	}

	return totalVotes
}

// Train updates the Tsetlin Machine with a new training example
func (tm *TsetlinMachine) Train(input []float64, targetClass int) {
	prediction := tm.Predict(input)
	if tm.debug {
		log.Printf("Training: Input=%v, Target Class=%d, Predicted Class=%d", input, targetClass, prediction.PredictedClass)
	}

	if prediction.PredictedClass == targetClass {
		if tm.debug {
			log.Printf("Correct prediction. Applying Type I feedback for class %d", targetClass)
		}
		tm.updateTypeI(input, targetClass)
	} else {
		if tm.debug {
			log.Printf("Incorrect prediction. Applying Type I feedback for target class %d and Type II feedback for predicted class %d", targetClass, prediction.PredictedClass)
		}
		tm.updateTypeI(input, targetClass)
		tm.updateTypeII(input, prediction.PredictedClass)
	}

	// Update bitfields after training
	for class := 0; class < tm.numClasses; class++ {
		for clause := 0; clause < tm.numClauses; clause++ {
			tm.updateBitSets(class, clause)
		}
	}
}

// Predict returns the prediction results including votes for each class and the predicted class
func (tm *TsetlinMachine) Predict(input []float64) PredictionResult {
	scores := make([]float64, tm.numClasses)
	var wg sync.WaitGroup
	var mu sync.Mutex
	semaphore := make(chan struct{}, tm.numWorkers)

	// Calculate scores for each class in parallel
	for class := 0; class < tm.numClasses; class++ {
		wg.Add(1)
		semaphore <- struct{}{}
		go func(class int) {
			defer wg.Done()
			defer func() { <-semaphore }()
			score := tm.calculateScore(input, class)
			mu.Lock()
			scores[class] = score
			mu.Unlock()
		}(class)
	}
	wg.Wait()

	// Find the class with the highest score
	maxScore := scores[0]
	predictedClass := 0
	secondHighestScore := float64(-1)

	for class := 1; class < tm.numClasses; class++ {
		if scores[class] > maxScore {
			secondHighestScore = maxScore
			maxScore = scores[class]
			predictedClass = class
		} else if scores[class] > secondHighestScore {
			secondHighestScore = scores[class]
		}
	}

	// Calculate margin and confidence using absolute values
	margin := math.Abs(maxScore - secondHighestScore)
	maxPossibleVotes := float64(tm.numClauses) / float64(tm.numClasses)
	confidence := math.Abs(margin / maxPossibleVotes)

	return PredictionResult{
		Votes:          scores,
		PredictedClass: predictedClass,
		Margin:         margin,
		Confidence:     confidence,
	}
}

// ClauseInfo represents information about a learned clause
type ClauseInfo struct {
	Class    int      // Class this clause belongs to
	Index    int      // Index of the clause
	Sign     int      // Sign of the clause (+1 or -1)
	Literals []string // String representation of literals
	IsActive bool     // Whether the clause is currently active
}

// GetClauseInfo returns information about all clauses for a given class
func (tm *TsetlinMachine) GetClauseInfo(class int) []ClauseInfo {
	clauses := make([]ClauseInfo, tm.numClauses)

	for clause := 0; clause < tm.numClauses; clause++ {
		info := ClauseInfo{
			Class:    class,
			Index:    clause,
			Sign:     tm.clauseSigns[class][clause],
			Literals: make([]string, tm.numLiterals),
		}

		// Get literal information
		for literal := 0; literal < tm.numLiterals; literal++ {
			idx := clause*tm.numLiterals + literal
			action := tm.automata[class][idx].getAction()

			if action == 1 {
				info.Literals[literal] = fmt.Sprintf("x%d", literal)
			} else {
				info.Literals[literal] = fmt.Sprintf("¬x%d", literal)
			}
		}

		clauses[clause] = info
	}

	return clauses
}

// GetActiveClauses returns information about currently active clauses for a given input
func (tm *TsetlinMachine) GetActiveClauses(input []float64, class int) []ClauseInfo {
	allClauses := tm.GetClauseInfo(class)
	activeClauses := make([]ClauseInfo, 0)

	for _, clause := range allClauses {
		clauseActive := true

		for literal := 0; literal < tm.numLiterals; literal++ {
			idx := clause.Index*tm.numLiterals + literal
			action := tm.automata[class][idx].getAction()

			if action == 1 {
				if input[literal] == 0 {
					clauseActive = false
					break
				}
			} else {
				if input[literal] == 1 {
					clauseActive = false
					break
				}
			}
		}

		if clauseActive {
			clause.IsActive = true
			activeClauses = append(activeClauses, clause)
		}
	}

	return activeClauses
}

// String returns a string representation of a clause
func (ci ClauseInfo) String() string {
	literalsStr := strings.Join(ci.Literals, " ∧ ")
	sign := "+"
	if ci.Sign < 0 {
		sign = "-"
	}
	return fmt.Sprintf("Class %d, Clause %d (%s): %s", ci.Class, ci.Index, sign, literalsStr)
}

// updateTypeI implements Type I feedback (reward) using global worker pool
func (tm *TsetlinMachine) updateTypeI(input []float64, class int) {
	var wg sync.WaitGroup
	wg.Add(tm.numClauses)

	for clause := 0; clause < tm.numClauses; clause++ {
		clause := clause // Create new variable for goroutine
		globalWorkerPool.Submit(func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(clause)))

			for literal := 0; literal < tm.numLiterals; literal++ {
				idx := clause*tm.numLiterals + literal
				if rng.Float64() < tm.s/(tm.s+1) {
					if input[literal] > 0.5 {
						tm.mu.Lock()
						tm.automata[class][idx].reward()
						if tm.debug {
							log.Printf("Type I feedback: Class=%d, Clause=%d, Literal=%d, State=%d", class, clause, literal, tm.automata[class][idx].state)
						}
						tm.mu.Unlock()
					}
				}
			}
		})
	}
	wg.Wait()
}

// updateTypeII implements Type II feedback (penalty) using global worker pool
func (tm *TsetlinMachine) updateTypeII(input []float64, class int) {
	var wg sync.WaitGroup
	wg.Add(tm.numClauses)

	for clause := 0; clause < tm.numClauses; clause++ {
		clause := clause // Create new variable for goroutine
		globalWorkerPool.Submit(func() {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(clause)))

			for literal := 0; literal < tm.numLiterals; literal++ {
				idx := clause*tm.numLiterals + literal
				if rng.Float64() < 1/(tm.s+1) {
					if input[literal] > 0.5 {
						tm.mu.Lock()
						tm.automata[class][idx].penalty()
						if tm.debug {
							log.Printf("Type II feedback: Class=%d, Clause=%d, Literal=%d, State=%d", class, clause, literal, tm.automata[class][idx].state)
						}
						tm.mu.Unlock()
					}
				}
			}
		})
	}
	wg.Wait()
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

// Fit trains the Tsetlin Machine on the provided data
func (tm *TsetlinMachine) Fit(X [][]float64, y []int, epochs int) {
	if len(X) != len(y) {
		panic("X and y must have the same length")
	}

	// Set training mode
	tm.isTraining = true
	defer func() { tm.isTraining = false }()

	// Create a slice of indices for shuffling
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Shuffle the data
		tm.rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})

		// Process samples in parallel using worker pool
		var wg sync.WaitGroup
		semaphore := make(chan struct{}, tm.numWorkers)

		for _, idx := range indices {
			wg.Add(1)
			semaphore <- struct{}{} // Acquire semaphore
			go func(idx int) {
				defer wg.Done()
				defer func() { <-semaphore }() // Release semaphore
				tm.Train(X[idx], y[idx])
			}(idx)
		}
		wg.Wait()
	}
}
