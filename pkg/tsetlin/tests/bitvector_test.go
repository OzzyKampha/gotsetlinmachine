package tests

import (
	"testing"

	"github.com/OzzyKampha/gotsetlinmachine/pkg/tsetlin"
)

func TestNewBitVector(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		expected int
	}{
		{"zero size", 0, 0},
		{"small size", 10, 1},
		{"exact word size", 64, 1},
		{"multiple words", 100, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bv := tsetlin.NewBitVector(tt.size)
			if len(bv) != tt.expected {
				t.Errorf("NewBitVector(%d) = %d words, want %d", tt.size, len(bv), tt.expected)
			}
		})
	}
}

func TestBitVectorSetGet(t *testing.T) {
	bv := tsetlin.NewBitVector(100)
	tests := []struct {
		name     string
		index    int
		expected bool
	}{
		{"set first bit", 0, true},
		{"set last bit", 99, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bv.Set(tt.index)
			if got := bv.Get(tt.index); got != tt.expected {
				t.Errorf("Get(%d) = %v, want %v", tt.index, got, tt.expected)
			}
		})
	}
}

func TestBitVectorIsSubset(t *testing.T) {
	tests := []struct {
		name     string
		a        []uint64
		b        []uint64
		expected bool
	}{
		{
			"empty vectors",
			[]uint64{},
			[]uint64{},
			true,
		},
		{
			"a is subset",
			[]uint64{0b1010},
			[]uint64{0b1111},
			true,
		},
		{
			"a is not subset",
			[]uint64{0b1010},
			[]uint64{0b0101},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := tsetlin.BitVector(tt.a)
			b := tsetlin.BitVector(tt.b)
			if got := a.IsSubset(b); got != tt.expected {
				t.Errorf("IsSubset() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPackInputVector(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		expected tsetlin.BitVector
	}{
		{
			"empty input",
			[]int{},
			tsetlin.BitVector{},
		},
		{
			"single word",
			[]int{1, 0, 1, 0},
			tsetlin.BitVector{0b0101},
		},
		{
			"multiple words",
			[]int{1, 0, 1, 0, 1, 0, 1, 0, 1},
			tsetlin.BitVector{0b0101010101},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tsetlin.PackInputVector(tt.input)
			if len(got) != len(tt.expected) {
				t.Errorf("PackInputVector() = %v words, want %v", len(got), len(tt.expected))
				return
			}
			for i := range got {
				if got[i] != tt.expected[i] {
					t.Errorf("PackInputVector()[%d] = %b, want %b", i, got[i], tt.expected[i])
				}
			}
		})
	}
}
