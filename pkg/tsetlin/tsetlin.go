package tsetlin

import (
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

// EvaluateClause determines if a clause is satisfied by the input.
// A clause is satisfied if all its included literals are present and all its
// excluded literals are absent in the input.
func EvaluateClause(c Clause, input BitVector) bool {
	if rand.Float32() < c.DropoutProb {
		return false
	}
	for w := 0; w < len(input); w++ {
		word := input[w]
		if word == 0 {
			continue
		}
		for bit := 0; bit < wordSize; bit++ {
			if (word>>bit)&1 == 1 {
				idx := w*wordSize + bit
				if idx < len(c.Exclude)*4 && c.Exclude.Get(idx) >= ActivationThreshold {
					return false
				}
			}
		}
	}
	for w := 0; w < len(input); w++ {
		notWord := ^input[w]
		if notWord == 0 {
			continue
		}
		for bit := 0; bit < wordSize; bit++ {
			if (notWord>>bit)&1 == 1 {
				idx := w*wordSize + bit
				if idx < len(c.Include)*4 && c.Include.Get(idx) >= ActivationThreshold {
					return false
				}
			}
		}
	}
	return true
}

// typeIFeedback applies type I feedback to a clause.
// Type I feedback is used to reinforce correct predictions by strengthening
// the association between literals and their clauses.
func typeIFeedback(clause *Clause, input BitVector, s int) {
	for w := 0; w < len(input); w++ {
		word := input[w]
		for bit := 0; bit < wordSize; bit++ {
			idx := w*wordSize + bit
			if idx < len(clause.Include)*4 {
				if (word>>bit)&1 == 1 {
					if rand.Float32() < 1.0/float32(s) {
						clause.Include.Inc(idx)
						clause.Exclude.Dec(idx)
					}
				} else {
					if rand.Float32() < 1.0/float32(s) {
						clause.Exclude.Inc(idx)
						clause.Include.Dec(idx)
					}
				}
			}
		}
	}
}

// typeIIFeedback applies type II feedback to a clause.
// Type II feedback is used to correct incorrect predictions by weakening
// the association between literals and their clauses.
func typeIIFeedback(clause *Clause, input BitVector, s int) {
	for w := 0; w < len(input); w++ {
		word := input[w]
		for bit := 0; bit < wordSize; bit++ {
			idx := w*wordSize + bit
			if idx < len(clause.Include)*4 {
				if (word>>bit)&1 == 1 {
					if rand.Float32() < 1.0/float32(s) {
						clause.Include.Dec(idx)
						clause.Exclude.Dec(idx)
					}
				}
			}
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
func (tm *TsetlinMachine) Predict(X interface{}, numWorkers int) interface{} {
	var inputs [][]int
	switch x := X.(type) {
	case []int:
		// Convert single input to batch of size 1
		inputs = [][]int{x}
	case [][]int:
		inputs = x
	default:
		panic("Predict expects either []int or [][]int")
	}

	n := len(inputs)
	results := make([]int, n)
	jobs := make(chan int, n)
	var wg sync.WaitGroup

	// Use CPU count if numWorkers is 0
	if numWorkers == 0 {
		numWorkers = runtime.NumCPU()
	}

	worker := func() {
		for i := range jobs {
			bv := PackInputVector(inputs[i])
			sum := 0.0
			for _, c := range tm.Clauses {
				if EvaluateClause(c, bv) {
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

	// Return single result for single input
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
						clause.Weight += 0.01
						if clause.Weight > 3.0 {
							clause.Weight = 3.0
						}
					}
				} else if fType == -1 {
					typeIIFeedback(clause, bv, tm.S)
					// Decay clause if it misfired
					if EvaluateClause(*clause, bv) {
						clause.Weight -= 0.01
						if clause.Weight < 0.1 {
							clause.Weight = 0.1
						}
					}
				}
			}
		}
	}
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
	}
	for i := 0; i < numClasses; i++ {
		m.Classes[i] = NewTsetlinMachine(numClauses, numFeatures, threshold, s)
	}
	return m
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
