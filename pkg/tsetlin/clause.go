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
