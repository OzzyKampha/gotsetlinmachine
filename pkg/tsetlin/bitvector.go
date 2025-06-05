package tsetlin

// NewBitVector creates a new BitVector with the specified size.
// The size is rounded up to the nearest multiple of wordSize for efficient storage.
func NewBitVector(size int) BitVector {
	return make(BitVector, (size+wordSize-1)/wordSize)
}

// Set sets the bit at the specified index to 1.
// The index is automatically mapped to the correct word and bit position.
func (bv BitVector) Set(idx int) {
	word := idx / wordSize
	bit := idx % wordSize
	bv[word] |= 1 << bit
}

// Get returns the value of the bit at the specified index.
// Returns true if the bit is 1, false if it is 0.
func (bv BitVector) Get(idx int) bool {
	word := idx / wordSize
	bit := idx % wordSize
	return (bv[word]>>bit)&1 == 1
}

// IsSubset checks if the current BitVector is a subset of another BitVector.
// A BitVector A is a subset of B if all bits set in A are also set in B.
func (bv BitVector) IsSubset(other BitVector) bool {
	for i := range bv {
		if bv[i]&^other[i] != 0 {
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

// PackInputVector converts a slice of integers into a BitVector.
// Each integer in the input slice is treated as a binary value (0 or 1).
// The resulting BitVector is more memory efficient and faster to process.
func PackInputVector(input []int) BitVector {
	bv := NewBitVector(len(input))
	for i, v := range input {
		if v == 1 {
			bv.Set(i)
		}
	}
	return bv
}
