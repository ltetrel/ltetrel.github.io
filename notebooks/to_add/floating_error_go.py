'''
---
jupyter:
  kernelspec:
    display_name: gophernotes
    language: go
    name: gophernotes
---
'''
# %% [markdown]
'''
# Discovering Golang by showing floating point operations errors
'''
# %% [markdown]
'''
## tl;dr
1. Conventionnally, numbers are stored in the machine up-to 64bit.
2. Numerical operations cannot have infinite precision because of bounded memory space
'''
# %% [markdown]
'''
Recently I have been playing with Golang. 
This new language was designed in 2007 by Google engineers.
Their servers was having more and more trouble to manage the growing web requests, so they created [Go](https://golang.org/)
which makes it easier to build simple, reliable, and efficient software.
Here is the list of some advantages and disadvantages over Python (taken completly randomly :D):
+ Built-in concurrency (and hence also multi-processing)
+ Fast compiled language (faster than C++)
+ More reproducible (hail to deep learning in python!!)
- Harder to learn
- Lack of libraries
I was specially hyped with the built-in concurrency feature, after suffering a lot from multi-processing in python when I tried for example to implement a
[neuroimaging data grabber for deep learning](https://github.com/SIMEXP/DeepNeuroAN/blob/master/deepneuroan/data_generator.py).
This is mostly due to the global interpreter lock, which bound python processes to one core.
(check [this](http://python-notes.curiousefficiency.org/en/latest/python3/multicore_python.html#why-is-using-a-global-interpreter-lock-gil-a-problem) "problem").
'''
# %% [markdown]
'''
In this post I will feature Go by showing how cpu numerical restrictions can impact your calculations.
To install gophernotes, the official golang kernel for jupyter, I followed the instructions [here](https://github.com/gopherdata/gophernotes#linux).
'''

# %%
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
	// At one point, the addition is equal to a rounded number. The addition to add is so low
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