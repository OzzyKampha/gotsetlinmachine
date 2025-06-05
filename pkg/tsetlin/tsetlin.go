// tsetlin_machine_clause_skipping.go
package tsetlin

import (
	"math/rand"
	"sync"
)

const (
	wordSize            = 64
	T                   = 100                // Used for prediction decision
	stateMax            = 199                // Defines automaton state range [0, stateMax]
	ActivationThreshold = (stateMax + 1) / 2 // Used for automaton activation threshold
)

type BitVector []uint64

func NewBitVector(size int) BitVector {
	return make(BitVector, (size+wordSize-1)/wordSize)
}

func (bv BitVector) Set(idx int) {
	bv[idx/wordSize] |= 1 << (idx % wordSize)
}

func (bv BitVector) Get(idx int) bool {
	return (bv[idx/wordSize] & (1 << (idx % wordSize))) != 0
}

func (bv BitVector) IsSubsetOf(other BitVector) bool {
	for i := 0; i < len(bv); i++ {
		if (bv[i] & other[i]) != bv[i] {
			return false
		}
	}
	return true
}

func (bv BitVector) InvertedSubsetOf(other BitVector) bool {
	for i := 0; i < len(bv); i++ {
		if (^other[i] & bv[i]) != bv[i] {
			return false
		}
	}
	return true
}

type PackedStates []uint64

func NewPackedStates(size int) PackedStates {
	return make(PackedStates, (size*16+63)/64) // 16 bits per state
}

func (ps PackedStates) get(i int) uint16 {
	word := (i * 16) / 64
	shift := (i * 16) % 64
	return uint16((ps[word] >> shift) & 0xFFFF)
}

func (ps PackedStates) set(i int, val uint16) {
	word := (i * 16) / 64
	shift := (i * 16) % 64
	ps[word] &^= 0xFFFF << shift
	ps[word] |= uint64(val) << shift
}

func (ps PackedStates) inc(i int) {
	val := ps.get(i)
	if val < stateMax {
		ps.set(i, val+1)
	}
}

func (ps PackedStates) dec(i int) {
	val := ps.get(i)
	if val > 0 {
		ps.set(i, val-1)
	}
}

type Clause struct {
	include     PackedStates
	exclude     PackedStates
	Vote        int
	Weight      float32
	dropoutProb float32
}

type TsetlinMachine struct {
	Clauses       []Clause
	NumFeatures   int
	VoteThreshold int
	S             int
}

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
			include.set(j, uint16(val))
			exclude.set(j, uint16(val))
		}
		tm.Clauses[i] = Clause{
			include:     include,
			exclude:     exclude,
			Vote:        1 - 2*(i%2),
			Weight:      1.0,
			dropoutProb: 0.0,
		}
	}
	return tm
}

func PackInputVector(input []int) BitVector {
	bv := NewBitVector(len(input))
	for i, bit := range input {
		if bit != 0 {
			bv.Set(i)
		}
	}
	return bv
}

func EvaluateClause(c Clause, input BitVector) bool {
	if rand.Float32() < c.dropoutProb {
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
				if idx < len(c.exclude)*4 && c.exclude.get(idx) >= ActivationThreshold {
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
				if idx < len(c.include)*4 && c.include.get(idx) >= ActivationThreshold {
					return false
				}
			}
		}
	}
	return true
}

func typeIFeedback(clause *Clause, input BitVector, s int) {
	for w := 0; w < len(input); w++ {
		word := input[w]
		for bit := 0; bit < wordSize; bit++ {
			idx := w*wordSize + bit
			if idx < len(clause.include)*4 {
				if (word>>bit)&1 == 1 {
					if rand.Float32() < 1.0/float32(s) {
						clause.include.inc(idx)
						clause.exclude.dec(idx)
					}
				} else {
					if rand.Float32() < 1.0/float32(s) {
						clause.exclude.inc(idx)
						clause.include.dec(idx)
					}
				}
			}
		}
	}
}

func typeIIFeedback(clause *Clause, input BitVector, s int) {
	for w := 0; w < len(input); w++ {
		word := input[w]
		for bit := 0; bit < wordSize; bit++ {
			idx := w*wordSize + bit
			if idx < len(clause.include)*4 {
				if (word>>bit)&1 == 1 {
					if rand.Float32() < 1.0/float32(s) {
						clause.include.dec(idx)
						clause.exclude.dec(idx)
					}
				}
			}
		}
	}
}

func (tm *TsetlinMachine) Predict(input []int) int {
	bv := PackInputVector(input)
	sum := 0.0
	for _, c := range tm.Clauses {
		if EvaluateClause(c, bv) {
			sum += float64(c.Vote) * float64(c.Weight)
		}
	}
	if sum >= float64(tm.VoteThreshold) {
		return 1
	}
	return 0
}

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

func (tm *TsetlinMachine) Fit(X [][]int, Y []int, targetClass int, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range X {
			input := X[i]
			y := 0
			if Y[i] == targetClass {
				y = 1
			}
			prediction := tm.Predict(input)
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

type MultiClassTM struct {
	Classes []*TsetlinMachine
}

func NewMultiClassTM(numClasses, numClauses, numFeatures, threshold, s int) *MultiClassTM {
	m := &MultiClassTM{
		Classes: make([]*TsetlinMachine, numClasses),
	}
	for i := 0; i < numClasses; i++ {
		m.Classes[i] = NewTsetlinMachine(numClauses, numFeatures, threshold, s)
	}
	return m
}

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

func (m *MultiClassTM) Predict(X []int) int {
	bestClass := -1
	bestScore := -1 << 30
	for class, tm := range m.Classes {
		score := tm.Score(X)
		if score > bestScore {
			bestScore = score
			bestClass = class
		}
	}
	return bestClass
}
