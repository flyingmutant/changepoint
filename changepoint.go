// Copyright 2020 Gregory Petrosyan <gregory.petrosyan@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package changepoint implements algorithms for changepoint detection.
package changepoint

import (
	"math"
)

// Calculates the cost of the (tau1; tau2] segment.
// Remember that tau are one-based indexes.
type costFunc func(tau0 int, tau1 int, tau2 int) float64

func pelt(data []float64, minDistance int, cost costFunc, penalty float64) []int {
	n := len(data)

	// We will use dynamic programming to find the best solution; `bestCost` is the cost array.
	// `bestCost[i]` is the cost for subarray `data[0..i-1]`.
	// It's a 1-based array (`data[0]`..`data[n-1]` correspond to `bestCost[1]`..`bestCost[n]`)
	bestCost := make([]float64, n+1)
	bestCost[0] = -penalty
	for curTau := minDistance; curTau < 2*minDistance; curTau++ {
		bestCost[curTau] = cost(0, 0, curTau)
	}

	// `prevChangepointIndex` is an array of references to previous changepoints. If the current segment ends at
	// the position `i`, the previous segment ends at the position `prevChangepointIndex[i]`. It's a 1-based
	// array (`data[0]`..`data[n-1]` correspond to the `prevChangepointIndex[1]`..`prevChangepointIndex[n]`)
	prevChangepointIndex := make([]int, n+1)

	// We use PELT (Pruned Exact Linear Time) approach which means that instead of enumerating all possible previous
	// tau values, we use a whitelist of "good" tau values that can be used in the optimal solution. If we are 100%
	// sure that some of the tau values will not help us to form the optimal solution, such values should be
	// removed. See [Killick2012] for details.
	prevTaus := make([]int, n+1) // The maximum number of the previous tau values is n + 1
	prevTaus[0] = 0
	prevTaus[1] = minDistance
	costForPrevTau := make([]float64, n+1)
	prevTausCount := 2 // The counter of previous tau values. Defines the size of `prevTaus` and `costForPrevTau`.

	// Following the dynamic programming approach, we enumerate all tau positions. For each `curTau`, we pretend
	// that it's the end of the last segment and trying to find the end of the previous segment.
	for curTau := 2 * minDistance; curTau < n+1; curTau++ {
		// For each previous tau, we should calculate the cost of taking this tau as the end of the previous
		// segment. This cost equals the cost for the `prevTau` plus cost of the new segment (from `prevTau`
		// to `curTau`) plus penalty for the new changepoint.
		for i := 0; i < prevTausCount; i++ {
			prevTau := prevTaus[i]
			costForPrevTau[i] = bestCost[prevTau] +
				cost(prevChangepointIndex[prevTau], prevTau, curTau) +
				penalty
		}

		// Now we should choose the tau that provides the minimum possible cost.
		bestPrevTauIndex := whichMin(costForPrevTau[:prevTausCount])
		bestCost[curTau] = costForPrevTau[bestPrevTauIndex]
		prevChangepointIndex[curTau] = prevTaus[bestPrevTauIndex]

		// Prune phase: we remove "useless" tau values that will not help to achieve minimum cost in the future
		curBestCost := bestCost[curTau]
		newPrevTausCount := 0
		for i := 0; i < prevTausCount; i++ {
			if costForPrevTau[i] < curBestCost+penalty {
				prevTaus[newPrevTausCount] = prevTaus[i]
				newPrevTausCount++
			}
		}

		// We add a new tau value that is located on the `minDistance` distance from the next `curTau` value
		prevTaus[newPrevTausCount] = curTau - minDistance + 1
		prevTausCount = newPrevTausCount + 1
	}

	// Here we collect the result list of changepoint indexes `changepoints` using `prevChangepointIndex`
	var changepoints []int
	curIndex := prevChangepointIndex[n] // The index of the end of the last segment is `n`
	for curIndex != 0 {
		changepoints = append(changepoints, curIndex-1) // 1-based indexes should be be transformed to 0-based indexes
		curIndex = prevChangepointIndex[curIndex]
	}

	// The result changepoints should be sorted in ascending order.
	for left, right := 0, len(changepoints)-1; left < right; left, right = left+1, right-1 {
		changepoints[left], changepoints[right] = changepoints[right], changepoints[left]
	}

	return changepoints
}

func whichMin(data []float64) int {
	ix, min := -1, math.Inf(1)

	for i, v := range data {
		if v < min {
			ix, min = i, v
		}
	}

	return ix
}
