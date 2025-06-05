package tsetlin

// NewPackedStates creates a new PackedStates vector with the specified size.
// Each state is stored as a 16-bit value, packed into uint64 words for efficiency.
func NewPackedStates(size int) PackedStates {
	return make(PackedStates, (size+3)/4)
}

// get retrieves the state value at the specified index.
// The state value is a 16-bit unsigned integer representing the current state
// of a Tsetlin automaton.
func (ps PackedStates) get(idx int) uint16 {
	word := idx / 4
	shift := (idx % 4) * 16
	return uint16((ps[word] >> shift) & 0xFFFF)
}

// set updates the state value at the specified index.
// The state value is stored as a 16-bit unsigned integer within a uint64 word.
func (ps PackedStates) set(idx int, val uint16) {
	word := idx / 4
	shift := (idx % 4) * 16
	mask := uint64(0xFFFF) << shift
	ps[word] = (ps[word] &^ mask) | (uint64(val) << shift)
}

// inc increments the state value at the specified index if it's below stateMax.
// This is used to strengthen the association of a literal with its clause.
func (ps PackedStates) inc(idx int) {
	val := ps.get(idx)
	if val < stateMax {
		ps.set(idx, val+1)
	}
}

// dec decrements the state value at the specified index if it's above 0.
// This is used to weaken the association of a literal with its clause.
func (ps PackedStates) dec(idx int) {
	val := ps.get(idx)
	if val > 0 {
		ps.set(idx, val-1)
	}
}
