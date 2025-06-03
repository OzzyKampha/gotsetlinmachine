package tsetlin

import (
	"fmt"
	"math/rand"
)

// BitVec represents a bit vector using []uint64 for efficient bit operations.
type BitVec []uint64

// NewBitVec creates a new BitVec with the given number of bits.
func NewBitVec(numBits int) BitVec {
	words := (numBits + 63) / 64
	return make(BitVec, words)
}

// Set sets a bit at the given index.
func (b BitVec) Set(index int) {
	word, bit := index/64, index%64
	b[word] |= 1 << bit
}

// Clear clears a bit at the given index.
func (b BitVec) Clear(index int) {
	word, bit := index/64, index%64
	b[word] &^= 1 << bit
}

// Test returns whether a bit is set at the given index.
func (b BitVec) Test(index int) bool {
	word, bit := index/64, index%64
	return (b[word] & (1 << bit)) != 0
}

// BitPackedClause represents a clause using bit-packed include/exclude masks.
type BitPackedClause struct {
	IncludeMask BitVec
	ExcludeMask BitVec
	IsPositive  bool
	NumFeatures int
}

// NewBitPackedClause creates a new BitPackedClause with the given number of features.
func NewBitPackedClause(numFeatures int) *BitPackedClause {
	return &BitPackedClause{
		IncludeMask: NewBitVec(numFeatures),
		ExcludeMask: NewBitVec(numFeatures),
		NumFeatures: numFeatures,
		IsPositive:  true,
	}
}

// SetInclude sets whether a feature is included in the clause.
func (c *BitPackedClause) SetInclude(index int, included bool) {
	if included {
		c.IncludeMask.Set(index)
	} else {
		c.IncludeMask.Clear(index)
	}
}

// SetExclude sets whether a feature is excluded in the clause.
func (c *BitPackedClause) SetExclude(index int, excluded bool) {
	if excluded {
		c.ExcludeMask.Set(index)
	} else {
		c.ExcludeMask.Clear(index)
	}
}

// HasInclude returns whether a feature is included in the clause.
func (c *BitPackedClause) HasInclude(index int) bool {
	return c.IncludeMask.Test(index)
}

// HasExclude returns whether a feature is excluded in the clause.
func (c *BitPackedClause) HasExclude(index int) bool {
	return c.ExcludeMask.Test(index)
}

// Match checks if the clause matches the given input pattern.
func (c *BitPackedClause) Match(input BitVec) bool {
	for i := 0; i < len(input); i++ {
		if (input[i] & c.IncludeMask[i]) != c.IncludeMask[i] {
			return false
		}
		if (input[i] & c.ExcludeMask[i]) != 0 {
			return false
		}
	}
	return true
}

// FromFloat64Slice converts a float64 slice to a bit-packed BitVec.
func FromFloat64Slice(input []float64) BitVec {
	bv := NewBitVec(len(input))
	for i, v := range input {
		if v != 0 {
			bv.Set(i)
		}
	}
	return bv
}

// ToFloat64Slice converts a bit-packed BitVec back to a float64 slice.
func ToFloat64Slice(input BitVec, length int) []float64 {
	result := make([]float64, length)
	for i := 0; i < length; i++ {
		if input.Test(i) {
			result[i] = 1.0
		}
	}
	return result
}

// BitPackedTsetlinMachine represents a Tsetlin Machine using bit-packed operations
// for efficient storage and processing of clauses.
type BitPackedTsetlinMachine struct {
	Clauses     []*BitPackedClause
	NumFeatures int
	NumClauses  int
	Threshold   float64
	S           float64
}

// NewBitPackedTsetlinMachine creates a new BitPackedTsetlinMachine with the given configuration.
func NewBitPackedTsetlinMachine(config Config) *BitPackedTsetlinMachine {
	tm := &BitPackedTsetlinMachine{
		NumFeatures: config.NumFeatures,
		NumClauses:  config.NumClauses,
		Threshold:   config.Threshold,
		S:           config.S,
		Clauses:     make([]*BitPackedClause, config.NumClauses),
	}

	for i := range tm.Clauses {
		tm.Clauses[i] = NewBitPackedClause(config.NumFeatures)
		// Ensure each clause has at least one active include literal
		tm.Clauses[i].SetInclude(i%config.NumFeatures, true)
	}

	return tm
}

// PredictBitVec returns the prediction for the given bit-packed input pattern.
func (tm *BitPackedTsetlinMachine) PredictBitVec(input BitVec) int {
	sum := 0
	for _, clause := range tm.Clauses {
		if clause.Match(input) {
			sum++
		}
	}
	if float64(sum) >= tm.Threshold {
		return 1
	}
	return 0
}

// Predict returns the prediction for the given input pattern (user-friendly API).
func (tm *BitPackedTsetlinMachine) Predict(input []float64) int {
	bitInput := FromFloat64Slice(input)
	return tm.PredictBitVec(bitInput)
}

// UpdateBitVec updates the machine's state based on the bit-packed input pattern and target.
func (tm *BitPackedTsetlinMachine) UpdateBitVec(input BitVec, target int) {
	prediction := tm.PredictBitVec(input)
	for _, clause := range tm.Clauses {
		matches := clause.Match(input)
		if prediction != target {
			if matches {
				for i := 0; i < tm.NumFeatures; i++ {
					if clause.HasInclude(i) && input.Test(i) {
						if rand.Float64() < 1.0/tm.S {
							clause.SetInclude(i, false)
						}
					}
				}
			} else {
				for i := 0; i < tm.NumFeatures; i++ {
					if !clause.HasInclude(i) && input.Test(i) {
						if rand.Float64() < 1.0/tm.S {
							clause.SetInclude(i, true)
						}
					}
				}
			}
		}
	}
}

// Update updates the machine's state based on the input pattern and target (user-friendly API).
func (tm *BitPackedTsetlinMachine) Update(input []float64, target int) {
	bitInput := FromFloat64Slice(input)
	tm.UpdateBitVec(bitInput, target)
}

// GetClauseLiterals returns the include mask for a given clause (for interpretability).
func (tm *BitPackedTsetlinMachine) GetClauseLiterals(clauseIndex int) ([]bool, error) {
	if clauseIndex < 0 || clauseIndex >= len(tm.Clauses) {
		return nil, fmt.Errorf("invalid clause index: %d", clauseIndex)
	}
	clause := tm.Clauses[clauseIndex]
	literals := make([]bool, tm.NumFeatures)
	for i := 0; i < tm.NumFeatures; i++ {
		literals[i] = clause.HasInclude(i)
	}
	return literals, nil
}

type BitPackedClauseInfo struct {
	IncludeMask []bool
	ExcludeMask []bool
	IsPositive  bool
}

// GetClauseInfo returns information about all clauses in the machine.
func (tm *BitPackedTsetlinMachine) GetClauseInfo() []BitPackedClauseInfo {
	info := make([]BitPackedClauseInfo, len(tm.Clauses))
	for i, clause := range tm.Clauses {
		include := make([]bool, tm.NumFeatures)
		exclude := make([]bool, tm.NumFeatures)
		for j := 0; j < tm.NumFeatures; j++ {
			include[j] = clause.HasInclude(j)
			exclude[j] = clause.HasExclude(j)
		}
		info[i] = BitPackedClauseInfo{
			IncludeMask: include,
			ExcludeMask: exclude,
			IsPositive:  clause.IsPositive,
		}
	}
	return info
}

// GetActiveClauses returns the indices of clauses that match the given input.
func (tm *BitPackedTsetlinMachine) GetActiveClauses(input []float64) []int {
	bitInput := FromFloat64Slice(input)
	active := []int{}
	for i, clause := range tm.Clauses {
		if clause.Match(bitInput) {
			active = append(active, i)
		}
	}
	return active
}
