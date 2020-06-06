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

package changepoint

import (
	"math"
	"sort"
)

// NonParametric returns indexes of elements that split data into
// "statistically homogeneous" segments. NonParametric supports
// nonparametric distributions and has O(N*log(N)) algorithmic complexity.
// NonParametric uses ED-PELT algorithm for changepoint detection.
//
// The implementation is based on the following papers:
//
//   [Haynes2017] Kaylea Haynes, Paul Fearnhead, and Idris A. Eckley.
//   "A computationally efficient nonparametric approach for changepoint detection."
//   Statistics and Computing 27, no. 5 (2017): 1293-1305.
//   https://doi.org/10.1007/s11222-016-9687-5
//
//   [Killick2012] Rebecca Killick, Paul Fearnhead, and Idris A. Eckley.
//   "Optimal detection of changepoints with a linear computational cost."
//   Journal of the American Statistical Association 107, no. 500 (2012): 1590-1598.
//   https://arxiv.org/pdf/1101.1438.pdf
func NonParametric(data []float64, minDistance int) []int {
	if minDistance < 1 {
		panic("minDistance must be positive")
	}

	if len(data) <= 2 || len(data) < 2*minDistance {
		return nil
	}

	s := newEDState(data)

	return pelt(data, minDistance, s.cost, s.penalty)
}

type edState struct {
	n           int
	penalty     float64
	k           int
	partialSums []int
}

func newEDState(data []float64) *edState {
	n := len(data)

	// The penalty which we add to the final cost for each additional changepoint
	// Here we use the Modified Bayesian Information Criterion
	penalty := 3 * math.Log(float64(n))

	// `k` is the number of quantiles that we use to approximate an integral during the segment cost evaluation
	// We use `k=Ceiling(4*log(n))` as suggested in the Section 4.3 "Choice of K in ED-PELT" in [Haynes2017]
	// `k` can't be greater than `n`, so we should always use the `Min` function here (important for n <= 8)
	k := int(math.Min(float64(n), math.Ceil(4*math.Log(float64(n)))))

	// We should precalculate sums for empirical CDF, it will allow fast evaluating of the segment cost
	partialSums := calcPartialSums(data, k)

	return &edState{
		n:           n,
		penalty:     penalty,
		k:           k,
		partialSums: partialSums,
	}
}

// Partial sums for empirical CDF (formula (2.1) from Section 2.1 "Model" in [Haynes2017])
//
//   partialSums'[i, tau] = (count(data[j] < t) * 2 + count(data[j] == t) * 1) for j=0..tau-1
//   where t is the i-th quantile value (see Section 3.1 "Discrete approximation" in [Haynes2017] for details)
//
// In order to get better performance, we present
// a two-dimensional array partialSums'[k, n + 1] as a single-dimensional array partialSums[k * (n + 1)].
// We assume that partialSums'[i, tau] = partialSums[i * (n + 1) + tau].
//
// - We use doubled sum values in order to use []int instead of []float64 (it provides noticeable
//   performance boost). Thus, multipliers for count(data[j] < t) and count(data[j] == t) are
//   2 and 1 instead of 1 and 0.5 from the [Haynes2017].
// - Note that these quantiles are not uniformly distributed: tails of the data distribution contain more
//   quantile values than the center of the distribution
func calcPartialSums(data []float64, k int) []int {
	n := len(data)
	partialSums := make([]int, k*(n+1))
	sortedData := append([]float64(nil), data...)
	sort.Float64s(sortedData)

	offset := 0
	for i := 0; i < k; i++ {
		z := -1 + (2*float64(i)+1)/float64(k)       // Values from (-1+1/k) to (1-1/k) with step = 2/k
		p := 1 / (1 + math.Pow(2*float64(n)-1, -z)) // Values from 0.0 to 1.0
		t := sortedData[int(p*float64(n-1))]        // Quantile value, formula (2.1) in [Haynes2017]

		for tau := 1; tau <= n; tau++ {
			// `curPartialSumsValue` is a temp variable to keep the future value of `partialSums[offset + tau]`
			// (or `partialSums'[i, tau]`)
			curPartialSumsValue := partialSums[offset+tau-1]
			if data[tau-1] < t {
				curPartialSumsValue += 2 // We use doubled value (2) instead of original 1.0
			} else if data[tau-1] == t {
				curPartialSumsValue += 1 // We use doubled value (1) instead of original 0.5
			}

			partialSums[offset+tau] = curPartialSumsValue
		}

		offset += n + 1
	}

	return partialSums
}

func (ed *edState) cost(_ /* tau0 */ int, tau1 int, tau2 int) float64 {
	sum := 0.0
	offset := tau1 // offset of partialSums'[i, tau1] in the single-dimensional `partialSums` array
	tauDiff := tau2 - tau1
	for i := 0; i < ed.k; i++ {
		// actualSum is (count(data[j] < t) * 2 + count(data[j] == t) * 1) for j=tau1..tau2-1
		actualSum := ed.partialSums[offset+tauDiff] -
			ed.partialSums[offset] // partialSums'[i, tau2] - partialSums'[i, tau1]

		// We skip these two cases (correspond to fit = 0 or fit = 1) because of invalid math.Log values
		if actualSum != 0 && actualSum != tauDiff*2 {
			// Empirical CDF $\hat{F}_i(t)$ (Section 2.1 "Model" in [Haynes2017])
			fit := float64(actualSum) * 0.5 / float64(tauDiff)

			// Segment cost $\mathcal{L}_{np}$ (Section 2.2 "Nonparametric maximum likelihood" in [Haynes2017])
			lnp := float64(tauDiff) * (fit*math.Log(fit) + (1-fit)*math.Log(1-fit))
			sum += lnp
		}

		offset += ed.n + 1
	}

	c := -math.Log(2*float64(ed.n) - 1) // Constant from Lemma 3.1 in [Haynes2017]
	return 2 * c / float64(ed.k) * sum  // See Section 3.1 "Discrete approximation" in [Haynes2017]
}
