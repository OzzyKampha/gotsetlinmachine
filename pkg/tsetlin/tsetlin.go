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
func NewTsetlinMachine(numClauses, numFeatures, voteThreshold int, s float64) *TsetlinMachine {
	if voteThreshold == -1 {
		voteThreshold = 0
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
			DropoutProb: 0.2,
		}
	}
	return tm
}

// Score returns the weighted sum of clause votes for the input.
// This can be used to get a confidence score for the prediction.
func (tm *TsetlinMachine) Score(input []int) int {
	bv := PackInputVector(input)

	sum := 0.0
	for _, c := range tm.Clauses {
		if EvaluateClause(c, bv, false) {
			sum += float64(c.Vote) * float64(c.Weight)

		}
	}

	return int(sum)
}

// Predict makes predictions for inputs in parallel.
// If numWorkers is 0, it will use the number of available CPUs.
// Optimized parallel prediction with input vector pre-packing
func (tm *TsetlinMachine) Predict(X interface{}, numWorkers int, training bool) interface{} {
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
		numWorkers = runtime.NumCPU() * 2
	}

	worker := func() {
		for i := range jobs {
			sum := 0.0
			for _, c := range tm.Clauses {
				if EvaluateClause(c, packed[i], training) {
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
	batchSize := 32 // You can tune this
	total := len(X)

	// Dynamically size workers: 1 worker per 32 clauses, minimum 1
	numWorkers := len(tm.Clauses) / 32
	if numWorkers < 1 {
		numWorkers = 1
	} else if numWorkers > 32 {
		numWorkers = 32 // Cap to avoid oversubscription
	}

	tasks := make(chan ClauseUpdateTask, 1024)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go tm.updateClauseWorker(tasks, &wg)
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < total; b += batchSize {
			end := b + batchSize
			if end > total {
				end = total
			}

			for i := b; i < end; i++ {
				y := 0
				if Y[i] == targetClass {
					y = 1
				}

				input := X[i]
		for i := range X {
			input := X[i]
			y := 0
			if Y[i] == targetClass {
				y = 1
			}
			prediction := tm.Predict(input, 0, true).(int)
			feedback := y - prediction
			for j := range tm.Clauses {
				clause := &tm.Clauses[j]
				fType := feedback * clause.Vote
				bv := PackInputVector(input)
				prediction := tm.Predict(input, 0).(int)
				feedback := y - prediction

				for j := range tm.Clauses {
					tasks <- ClauseUpdateTask{
						ClauseIndex: j,
						Clause:      &tm.Clauses[j],
						Feedback:    feedback,
						Input:       bv,
						S:           tm.S,
					}
				}
			}
		}
	}

	close(tasks)
	wg.Wait()
}

func (tm *TsetlinMachine) updateClauseWorker(tasks <-chan ClauseUpdateTask, wg *sync.WaitGroup) {
	defer wg.Done()
	for task := range tasks {
		clause := task.Clause
		fType := task.Feedback * clause.Vote

		if fType == 1 {
			typeIFeedback(clause, task.Input, task.S)
			if EvaluateClause(*clause, task.Input) {
				updateClauseWeightPositive(clause)
				if clause.Weight > 3.0 {
					clause.Weight = 3.0
				}
			}
		} else if fType == -1 {
			typeIIFeedback(clause, task.Input, task.S)
			if EvaluateClause(*clause, task.Input) {
				updateClauseWeightNegative(clause)
				if clause.Weight < 0.1 {
					clause.Weight = 0.1
				if fType == 1 {
					typeIFeedback(clause, bv, tm.S)
					// Reinforce clause if it fired correctly
					if EvaluateClause(*clause, bv, true) {
						updateClauseWeightPositive(clause)
						if clause.Weight > 3.0 {
							clause.Weight = 3.0
						}
					}
				} else if fType == -1 {
					typeIIFeedback(clause, bv, tm.S)
					// Decay clause if it misfired
					if EvaluateClause(*clause, bv, true) {
						updateClauseWeightNegative(clause)
						if clause.Weight < 0.1 {
							clause.Weight = 0.1
						}
					}
				}
				clause.UpdateFeatureMask()
				//println(clause.FeatureMask)
			}
		}
	}
}}
