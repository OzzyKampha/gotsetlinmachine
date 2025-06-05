package tsetlin

import (
	"math/rand"
	"runtime"
	"testing"
)

// generateRandomData creates random input data and labels for benchmarking
func generateRandomData(numSamples, numFeatures int) ([][]int, []int) {
	X := make([][]int, numSamples)
	Y := make([]int, numSamples)
	for i := range X {
		X[i] = make([]int, numFeatures)
		for j := range X[i] {
			X[i][j] = rand.Intn(2)
		}
		Y[i] = rand.Intn(2)
	}
	return X, Y
}

// BenchmarkBitVectorOperations measures the performance of BitVector operations
func BenchmarkBitVectorOperations(b *testing.B) {
	b.Run("NewBitVector", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			NewBitVector(1000)
		}
	})

	b.Run("SetGet", func(b *testing.B) {
		bv := NewBitVector(1000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i % 1000
			bv.Set(idx)
			_ = bv.Get(idx)
		}
	})

	b.Run("IsSubset", func(b *testing.B) {
		a := NewBitVector(1000)
		bv := NewBitVector(1000)
		for i := 0; i < 500; i++ {
			a.Set(i)
			bv.Set(i)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = a.IsSubset(bv)
		}
	})

	b.Run("PackInputVector", func(b *testing.B) {
		input := make([]int, 1000)
		for i := range input {
			input[i] = rand.Intn(2)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = PackInputVector(input)
		}
	})
}

// BenchmarkPackedStatesOperations measures the performance of PackedStates operations
func BenchmarkPackedStatesOperations(b *testing.B) {
	b.Run("NewPackedStates", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			NewPackedStates(1000)
		}
	})

	b.Run("GetSet", func(b *testing.B) {
		ps := NewPackedStates(1000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i % 1000
			ps.Set(idx, uint16(i%65536))
			_ = ps.Get(idx)
		}
	})

	b.Run("IncDec", func(b *testing.B) {
		ps := NewPackedStates(1000)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			idx := i % 1000
			ps.Inc(idx)
			ps.Dec(idx)
		}
	})
}

// BenchmarkTsetlinMachineOperations measures the performance of Tsetlin Machine operations
func BenchmarkTsetlinMachineOperations(b *testing.B) {
	// Create a Tsetlin Machine with reasonable parameters
	tm := NewTsetlinMachine(100, 50, 50, 3)
	X, Y := generateRandomData(1000, 50)

	b.Run("Predict", func(b *testing.B) {
		input := X[0]
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tm.Predict(input)
		}
	})

	b.Run("Score", func(b *testing.B) {
		input := X[0]
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = tm.Score(input)
		}
	})

	b.Run("Fit", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tm.Fit(X, Y, 1, 1)
		}
	})

	b.Run("EvaluateClause", func(b *testing.B) {
		clause := tm.Clauses[0]
		input := PackInputVector(X[0])
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = EvaluateClause(clause, input)
		}
	})
}

// BenchmarkMultiClassTMOperations measures the performance of MultiClass TM operations
func BenchmarkMultiClassTMOperations(b *testing.B) {
	// Create a MultiClass TM with reasonable parameters
	m := NewMultiClassTM(3, 100, 50, 50, 3)
	X, Y := generateRandomData(1000, 50)

	b.Run("Predict", func(b *testing.B) {
		input := X[0]
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = m.Predict(input)
		}
	})

	b.Run("Fit", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.Fit(X, Y, 1)
		}
	})
}

func BenchmarkParallelPredict(b *testing.B) {
	tm := NewTsetlinMachine(100, 50, 50, 3)
	X, _ := generateRandomData(10000, 50)
	numWorkers := runtime.NumCPU()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.ParallelPredict(X, numWorkers)
	}
}
