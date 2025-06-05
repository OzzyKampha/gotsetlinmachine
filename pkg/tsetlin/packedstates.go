package tsetlin

// NewPackedStates creates a new PackedStates with the specified size.
// Each state is stored in 16 bits, so we need (size + 3) / 4 words.
func NewPackedStates(size int) PackedStates {
	return make(PackedStates, (size+3)/4)
}

// Get returns the state value at the specified index.
func (ps PackedStates) Get(idx int) uint16 {
	wordIdx := idx / 4
	bitOffset := (idx % 4) * 16
	return uint16((ps[wordIdx] >> bitOffset) & 0xFFFF)
}

// Set sets the state value at the specified index.
func (ps PackedStates) Set(idx int, val uint16) {
	wordIdx := idx / 4
	bitOffset := (idx % 4) * 16
	mask := uint64(0xFFFF) << bitOffset
	ps[wordIdx] = (ps[wordIdx] & ^mask) | (uint64(val) << bitOffset)
}

// Inc increments the state value at the specified index.
// The value is capped at stateMax.
func (ps PackedStates) Inc(idx int) {
	val := ps.Get(idx)
	if val < stateMax {
		ps.Set(idx, val+1)
	}
}

// Dec decrements the state value at the specified index.
// The value is capped at 0.
func (ps PackedStates) Dec(idx int) {
	val := ps.Get(idx)
	if val > 0 {
		ps.Set(idx, val-1)
	}
}
