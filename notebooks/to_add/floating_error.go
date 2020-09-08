package main

import (
	"fmt"
	"math"
)

func computeError(resMult float64) float64 {
	var error float64 = 0.
	error = 100 * math.Abs(resMult-1.0) / resMult

	return error
}

func equalToOneFloat64(limit float64, resMult float64) (float64, float64) {
	var error float64 = 0.

	for i := 0; i < int(limit); i++ {
		resMult += 1 / limit
	}
	error = computeError(resMult)

	return resMult, error
}

func equalToOneFloat32(limit float32, resMult float32) (float32, float64) {
	var error float64 = 0.

	for i := 0; i < int(limit); i++ {
		resMult += 1 / limit
		if i > int(1.677721e7) {
			fmt.Printf("idx: %d, current value: %.16f, addition: %.9f\n", i, resMult, 1/limit)
		}
	}
	error = computeError(float64(resMult))

	return resMult, error
}

func main() {
	//var limit32 float32 = 3.35544309e7
	// This is an interresting number, where the precision of float32 is unficient
	// 1at one point, the addition is equal to a rounded number. The addition to add is so low
	// that it cannot be added to this round number (more than 7 zeros after the virgule which appears to be the limit for float32).
	var limit32 float32 = 3.3554431e7
	// the error is even bigger (addition cannot be perfomed earlier in the loop) if limit is lower
	var limit32 float32 = 1e8
	var initMult32 float32 = 0.
	var limit64 float64 = float64(limit32)
	var initMult64 float64 = float64(initMult32)

	compMult32, error32 := equalToOneFloat32(limit32, initMult32)
	compMult64, error64 := equalToOneFloat64(limit64, initMult64)
	fmt.Printf("Computed float32 result: %.7f\n", compMult32)
	fmt.Printf("Error: %.9f%%\n", error32)
	fmt.Printf("Computed float64 result: %.16f\n", compMult64)
	fmt.Printf("Error: %.9f%%\n", error64)
}
