package main

import (
	"fmt"
)

func generateStratifiedSamples(totalSamples, numClasses, bitsPerPattern int) ([][]int, []int) {
	X := make([][]int, 0, totalSamples)
	y := make([]int, 0, totalSamples)

	samplesPerClass := totalSamples / numClasses
	for i := 0; i < numClasses; i++ {
		base := make([]int, bitsPerPattern)
		seed := uint32(2654435761 * uint32(i))
		for j := 0; j < bitsPerPattern; j++ {
			base[j] = int((seed >> (j % 24)) & 1)
		}

		for v := 0; v < samplesPerClass; v++ {
			variant := make([]int, bitsPerPattern)
			copy(variant, base)
			variant[(i*v*7+v)%bitsPerPattern] ^= 1
			X = append(X, variant)
			y = append(y, i)
		}
	}

	fmt.Printf("Generated %d stratified samples (%d per class, %d bits each)\n", len(X), samplesPerClass, bitsPerPattern)
	return X, y
}
