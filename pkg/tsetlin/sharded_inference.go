package tsetlin

import (
	"math"
	"sync"
)

// InferenceResult holds the results from a single worker's computation
type InferenceResult struct {
	Votes        []int
	MatchScores  []float64
	ClauseCounts []int
}

// ShardedInference performs inference by computing class scores in parallel, matching the standard multiclass inference logic
func ShardedInference(tm *MultiClassTsetlinMachine, X []float64) (int, float64, []float64, []float64) {
	numClasses := len(tm.machines)

	// Input validation: check input length
	if len(X) != tm.machines[0].config.NumFeatures {
		votes := make([]float64, numClasses)
		matchScores := make([]float64, numClasses)
		return -1, 0.0, votes, matchScores
	}

	votes := make([]float64, numClasses)
	matchScores := make([]float64, numClasses)
	clauseCounts := make([]int, numClasses)

	var wg sync.WaitGroup
	wg.Add(numClasses)

	for classIdx, machine := range tm.machines {
		go func(classIdx int, machine *TsetlinMachine) {
			defer wg.Done()

			// Convert input to bit pattern for clause skipping
			var inputFeatures uint64
			for i, val := range X {
				if val == 1 {
					inputFeatures |= 1 << i
				}
			}

			// Weighted voting from active clauses
			activeClauses := 0
			var avgMatchScore float64
			voteSum := 0.0
			for i := range machine.clauses {
				if !machine.canSkipClause(i, inputFeatures) {
					clauseOutput := machine.evaluateClause(X, machine.clauses[i])
					if clauseOutput == 1 {
						normScore := machine.matchScores[i] / (machine.matchScores[i] + 1.0)
						voteSum += normScore
						avgMatchScore += machine.matchScores[i]
						activeClauses++
					}
				}
			}

			// Set weighted votes
			votes[classIdx] = voteSum

			// Calculate average match score
			if activeClauses > 0 {
				matchScores[classIdx] = avgMatchScore / float64(activeClauses)
			}
			clauseCounts[classIdx] = activeClauses
		}(classIdx, machine)
	}

	wg.Wait()

	// Find the class with the highest votes
	maxVotes := votes[0]
	predictedClass := 0
	secondHighestVotes := math.Inf(-1)
	for i := 1; i < numClasses; i++ {
		if votes[i] > maxVotes {
			secondHighestVotes = maxVotes
			maxVotes = votes[i]
			predictedClass = i
		} else if votes[i] > secondHighestVotes {
			secondHighestVotes = votes[i]
		}
	}

	// Calculate confidence based on vote margin
	confidence := 0.0
	if maxVotes > 0 {
		margin := maxVotes - secondHighestVotes
		maxPossibleVotes := float64(tm.machines[0].config.NumClauses)
		confidence = margin / maxPossibleVotes
		if confidence > 1.0 {
			confidence = 1.0
		}
	}

	return predictedClass, confidence, votes, matchScores
}

// computeConfidence calculates the confidence score for the prediction
func computeConfidence(votes []float64, matchScores []float64, clauseCounts []int, predictedClass int) float64 {
	// Find second highest vote count
	secondMax := math.Inf(-1)
	maxVotes := votes[predictedClass]
	for i, v := range votes {
		if i != predictedClass && v > secondMax {
			secondMax = v
		}
	}

	// Calculate margin
	margin := maxVotes - secondMax
	maxPossibleVotes := float64(clauseCounts[predictedClass])
	if maxPossibleVotes == 0 {
		return 0.0
	}

	// Calculate average match score
	avgScore := 0.0
	if clauseCounts[predictedClass] > 0 {
		avgScore = matchScores[predictedClass] / float64(clauseCounts[predictedClass])
	}

	// Combine margin and average score with proper weighting
	confidence := 0.7*(margin/maxPossibleVotes) + 0.3*avgScore

	// Clamp between 0 and 1
	return math.Max(0.0, math.Min(1.0, confidence))
}
