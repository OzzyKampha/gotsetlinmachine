package tsetlin

import (
	"math"
	"math/bits"
	"math/rand"
)

func updateClauseWeightPositive(clause *Clause) {
	clause.Weight = math.Min(3.0, clause.Weight+0.01)
}

func updateClauseWeightNegative(clause *Clause) {
	clause.Weight = math.Max(0.1, clause.Weight-0.01)
}
func (c *Clause) UpdateFeatureMaskold() {
	size := len(c.Include)
	if len(c.Exclude) < size {
		size = len(c.Exclude)
	}

	if c.FeatureMask == nil || len(c.FeatureMask) != size {
		c.FeatureMask = make(BitVector, size)
	}

	for i := 0; i < size; i++ {
		c.FeatureMask[i] = c.Include[i] | c.Exclude[i]
	}
}

func (c *Clause) UpdateFeatureMask() {
	size := len(c.Include) // assuming same size for Exclude
	mask := make(BitVector, size)
	for i := range mask {
		mask[i] = c.Include[i] | c.Exclude[i]
	}
	c.FeatureMask = mask
}

// EvaluateClause determines if a clause is satisfied by the input.
// A clause is satisfied if all its included literals are present and all its
// excluded literals are absent in the input.
// Optimized clause evaluation with feature mask skipping
func EvaluateClause(c Clause, input BitVector, training bool) bool {
	if rand.Float32() < c.DropoutProb {
		return false
	}

	if !training && c.FeatureMask != nil && !ClauseIntersectsInput(c.FeatureMask, input) {
		println("Skip")
		return false
	}

	maxIdx := len(c.Exclude) * 4

	for w := 0; w < len(input); w++ {
		word := input[w]
		for word != 0 {
			bit := bits.TrailingZeros64(word)
			idx := w*wordSize + bit
			if idx < maxIdx && c.Exclude.Get(idx) >= ActivationThreshold {
				return false
			}
			word &= word - 1
		}

		notWord := ^input[w]
		for notWord != 0 {
			bit := bits.TrailingZeros64(notWord)
			idx := w*wordSize + bit
			if idx < maxIdx && c.Include.Get(idx) >= ActivationThreshold {
				return false
			}
			notWord &= notWord - 1
		}
	}

	return true
}

// ClauseIntersectsInput checks if clause feature mask overlaps with input
func ClauseIntersectsInput(mask, input BitVector) bool {
	for i := 0; i < len(mask) && i < len(input); i++ {
		if mask[i]&input[i] != 0 {
			return true
		}
	}
	return false
}

// typeIFeedback applies type I feedback to a clause.
// Type I feedback is used to reinforce correct predictions by strengthening
// the association between literals and their clauses.
func typeIFeedback(clause *Clause, input BitVector, s float64) {
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

func typeIIFeedback(clause *Clause, input BitVector, s float64) {
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
