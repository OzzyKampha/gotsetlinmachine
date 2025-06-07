package tsetlin

import (
	"math"
	"math/bits"
	"math/rand"
	"runtime"
	"sync"
)

// NewTsetlinMachine creates a new Tsetlin Machine with the specified parameters.
// numClauses: number of clauses in the machine
// numFeatures: number of input features
// voteThreshold: minimum number of votes required for positive prediction (-1 for default)
// s: specificity parameter that controls the probability of type I feedback
func NewTsetlinMachine(numClauses, numFeatures, voteThreshold, s int) *TsetlinMachine {
	if voteThreshold == -1 {
		voteThreshold = numClauses / 2
	}
	tm := &TsetlinMachine{
		Clauses:       make([]Clause, numClauses),
		NumFeatures:   numFeatures,
		VoteThreshold: voteThreshold,
		S:             s,
	}

	for i := range tm.Clauses {
		include := NewPackedStates(numFeatures)
		exclude := NewPackedStates(numFeatures)
		for j := 0; j < numFeatures; j++ {
			val := ActivationThreshold - 10 + rand.Intn(20)
			include.Set(j, uint16(val))
			exclude.Set(j, uint16(val))
		}
		tm.Clauses[i] = Clause{
			Include:     include,
			Exclude:     exclude,
			Vote:        1 - 2*(i%2),
			Weight:      1.0,
			DropoutProb: 0.0,
		}
	}
	return tm
}

func updateClauseWeightPositive(clause *Clause) {
	clause.Weight = math.Min(3.0, clause.Weight+0.01)
}

func updateClauseWeightNegative(clause *Clause) {
	clause.Weight = math.Max(0.1, clause.Weight-0.01)
}

// EvaluateClause determines if a clause is satisfied by the input.
// A clause is satisfied if all its included literals are present and all its
// excluded literals are absent in the input.
func EvaluateClause(c Clause, input BitVector) bool {
	if rand.Float32() < c.DropoutProb {
		return false
	}
	for w := 0; w < len(input); w++ {
		word := input[w]
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			idx := w*wordSize + bit
			if idx < len(c.Exclude)*4 && c.Exclude.Get(idx) >= ActivationThreshold {
				return false
			}
			word &= word - 1
		}
		notWord := ^input[w]
		for notWord != 0 {
			bit := bits.TrailingZeros64(notWord)
			idx := w*wordSize + bit
			if idx < len(c.Include)*4 && c.Include.Get(idx) >= ActivationThreshold {
				return false
			}
			notWord &= notWord - 1
		}
	}
	return true
}

// typeIFeedback applies type I feedback to a clause.
// Type I feedback is used to reinforce correct predictions by strengthening
// the association between literals and their clauses.
func typeIFeedback(clause *Clause, input BitVector, s int) {
	sInv := 1.0 / float32(s)
	for w := 0; w < len(input); w++ {
		word := input[w]
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			idx := w*wordSize + bit
			if idx < len(clause.Include)*4 && rand.Float32() < sInv {
				clause.Include.Inc(idx)
				clause.Exclude.Dec(idx)
			}
			word &= word - 1
		}
		notWord := ^input[w]
		for notWord != 0 {
			bit := bits.TrailingZeros64(notWord)
			idx := w*wordSize + bit
			if idx < len(clause.Exclude)*4 && rand.Float32() < sInv {
				clause.Exclude.Inc(idx)
				clause.Include.Dec(idx)
			}
			notWord &= notWord - 1
		}
	}
}

func typeIIFeedback(clause *Clause, input BitVector, s int) {
	sInv := 1.0 / float32(s)
	for w := 0; w < len(input); w++ {
		word := input[w]
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			idx := w*wordSize + bit
			if idx < len(clause.Include)*4 && rand.Float32() < sInv {
				clause.Include.Dec(idx)
				clause.Exclude.Dec(idx)
			}
			word &= word - 1
		}
	}
}

// Score returns the weighted sum of clause votes for the input.
// This can be used to get a confidence score for the prediction.
func (tm *TsetlinMachine) Score(input []int) int {
	bv := PackInputVector(input)
	sum := 0.0
	for _, c := range tm.Clauses {
		if EvaluateClause(c, bv) {
			sum += float64(c.Vote) * float64(c.Weight)
		}
	}
	return int(sum)
}

// Predict makes predictions for inputs in parallel.
// If numWorkers is 0, it will use the number of available CPUs.
// Optimized parallel prediction with input vector pre-packing
func (tm *TsetlinMachine) Predict(X interface{}, numWorkers int) interface{} {
	var inputs [][]int
	switch x := X.(type) {
	case []int:
		inputs = [][]int{x}
	case [][]int:
		inputs = x
	default:
		panic("Predict expects either []int or [][]int")
	}

	n := len(inputs)
	results := make([]int, n)
	packed := make([]BitVector, n)
	for i := 0; i < n; i++ {
		packed[i] = PackInputVector(inputs[i])
	}

	jobs := make(chan int, n)
	var wg sync.WaitGroup
	if numWorkers <= 0 {
		numWorkers = runtime.NumCPU()
	}

	worker := func() {
		for i := range jobs {
			sum := 0.0
			for _, c := range tm.Clauses {
				if EvaluateClause(c, packed[i]) {
					sum += float64(c.Vote) * float64(c.Weight)
				}
			}
			if sum >= float64(tm.VoteThreshold) {
				results[i] = 1
			} else {
				results[i] = 0
			}
		}
		wg.Done()
	}

	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go worker()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	if _, ok := X.([]int); ok {
		return results[0]
	}
	return results
}

// Fit trains the Tsetlin Machine on the provided data.
// X: input features
// Y: target labels
// targetClass: the class to learn (1 for positive, 0 for negative)
// epochs: number of training epochs
func (tm *TsetlinMachine) Fit(X [][]int, Y []int, targetClass int, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range X {
			input := X[i]
			y := 0
			if Y[i] == targetClass {
				y = 1
			}
			prediction := tm.Predict(input, 0).(int)
			feedback := y - prediction
			for j := range tm.Clauses {
				clause := &tm.Clauses[j]
				fType := feedback * clause.Vote
				bv := PackInputVector(input)

				if fType == 1 {
					typeIFeedback(clause, bv, tm.S)
					// Reinforce clause if it fired correctly
					if EvaluateClause(*clause, bv) {
						updateClauseWeightPositive(clause)
						if clause.Weight > 3.0 {
							clause.Weight = 3.0
						}
					}
				} else if fType == -1 {
					typeIIFeedback(clause, bv, tm.S)
					// Decay clause if it misfired
					if EvaluateClause(*clause, bv) {
						updateClauseWeightNegative(clause)
						if clause.Weight < 0.1 {
							clause.Weight = 0.1
						}
					}
				}
			}
		}
	}
}

// MultiClassTM represents a multiclass Tsetlin Machine
type MultiClassTM struct {
	Classes []*TsetlinMachine
	workers int
	jobChan chan int
	wg      sync.WaitGroup
}

// NewMultiClassTM creates a new multiclass Tsetlin Machine.
// numClasses: number of classes to classify
// numClauses: number of clauses per class
// numFeatures: number of input features
// threshold: vote threshold for each binary classifier
// s: specificity parameter
func NewMultiClassTM(numClasses, numClauses, numFeatures, threshold, s int) *MultiClassTM {
	m := &MultiClassTM{
		Classes: make([]*TsetlinMachine, numClasses),
		workers: runtime.NumCPU(),
		jobChan: make(chan int, 1000), // Buffer size of 1000 jobs
	}
	for i := 0; i < numClasses; i++ {
		m.Classes[i] = NewTsetlinMachine(numClauses, numFeatures, threshold, s)
	}
	return m
}

// PredictBatch performs batch prediction using a worker pool.
// It processes multiple inputs in parallel and returns their predictions.
func (m *MultiClassTM) PredictBatch(inputs [][]int) []int {
	numSamples := len(inputs)
	results := make([]int, numSamples)

	// Start workers if not already running
	if m.jobChan == nil {
		m.jobChan = make(chan int, 1000)
		m.startWorkers()
	}

	// Create a wait group for this batch
	var batchWg sync.WaitGroup
	batchWg.Add(numSamples)

	// Process each input
	for i := range inputs {
		go func(idx int) {
			defer batchWg.Done()
			results[idx] = m.Predict(inputs[idx])
		}(i)
	}

	// Wait for all predictions to complete
	batchWg.Wait()
	return results
}

// startWorkers initializes the worker pool
func (m *MultiClassTM) startWorkers() {
	for w := 0; w < m.workers; w++ {
		m.wg.Add(1)
		go func() {
			defer m.wg.Done()
			for range m.jobChan {
				// Workers are now used for batch processing
			}
		}()
	}
}

// Close stops all workers and cleans up resources
func (m *MultiClassTM) Close() {
	if m.jobChan != nil {
		close(m.jobChan)
		m.wg.Wait()
		m.jobChan = nil
	}
}

// Fit trains the multiclass Tsetlin Machine on the provided data.
// Training is performed in parallel for each class.
func (m *MultiClassTM) Fit(X [][]int, Y []int, epochs int) {
	var wg sync.WaitGroup
	for class := 0; class < len(m.Classes); class++ {
		wg.Add(1)
		go func(cls int) {
			defer wg.Done()
			m.Classes[cls].Fit(X, Y, cls, epochs)
		}(class)
	}
	wg.Wait()
}

// Predict makes a multiclass prediction for the input.
// Returns the class with the highest score.
func (m *MultiClassTM) Predict(X []int) int {
	bestClass := 0
	bestScore := m.Classes[0].Score(X)
	for class := 1; class < len(m.Classes); class++ {
		score := m.Classes[class].Score(X)
		if score > bestScore {
			bestScore = score
			bestClass = class
		}
	}
	return bestClass
}
