package transforms

import (
	"fmt"
	np "github.com/mdcfrancis/gonp"
	"github.com/pa-m/randomkit"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestConvolve1DProducesCorrectOutput(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5}
	kernel := []float64{1, 0, -1}
	convolved := Convolve1D(input, kernel)
	expected := []float64{-1, -2, -2, -2, -1}
	assert.Equal(t, expected, convolved)
}

func TestConvolve1DPanics(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5}
	kernel := []float64{1, 0, -1, 1}
	assert.Panics(t, func() { Convolve1D(input, kernel) })
}

func TestConvolve1DPanicsWhenKernelLengthToLarge(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5}
	kernel := []float64{1, 0, -1, 1, 0, 1, 0, 1, 1, 0, 1, 2}
	assert.Panics(t, func() { Convolve1D(input, kernel) })
}

func TestGaussianKernelGeneratesCorrectValues(t *testing.T) {
	truncation := 4.0
	sigma := 2.0
	radius := int(truncation*sigma + 0.5)
	fmt.Println("Radius", radius)
	kernel := _createGaussianKernel(sigma, radius)

	expect := np.NpArray{
		6.69162896e-05, 4.36349021e-04,
		2.21596317e-03, 8.76430436e-03,
		2.69959580e-02, 6.47599366e-02,
		1.20987490e-01, 1.76035759e-01,
		1.99474648e-01, 1.76035759e-01,
		1.20987490e-01, 6.47599366e-02,
		2.69959580e-02, 8.76430436e-03,
		2.21596317e-03, 4.36349021e-04,
		6.69162896e-05,
	}

	assert.Equal(t, 17, len(kernel))
	sum := 0.0
	for i, v := range kernel {
		sum += v
		assert.InDelta(t, expect[i], v, 0.0001)
	}
	assert.InDelta(t, 1.0, sum, 0.0001)
}

func TestGaussianFilterAppliesCorrectFilter(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	sigma := 2.0
	filtered := GaussianFilter1D(input, sigma)
	expect := []float64{
		2.16163332,
		2.53657354,
		3.2058847,
		4.04996302,
		4.95003698,
		5.7941153,
		6.46342646,
		6.83836668}
	for i, v := range filtered {
		assert.InDelta(t, expect[i], v, 0.0001)
	}
}

func TestGaussianFilter1D(t *testing.T) {
	rnd := randomkit.NewRandomkitSource(42)
	scale := 0.25
	size := 40
	noise := np.RandN(rnd, size).MulFloat64(scale)
	expected_noise := np.NpArray{
		0.12417854, -0.03456608, 0.16192213, 0.38075746,
		-0.05853834, -0.05853424, 0.3948032, 0.19185868,
		-0.1173686, 0.13564001, -0.11585442, -0.11643244,
		0.06049057, -0.47832006, -0.43122946, -0.14057188,
		-0.25320778, 0.07856183, -0.22700602, -0.35307593,
		0.36641219, -0.05644408, 0.01688205, -0.35618705,
		-0.13609568, 0.02773065, -0.28774839, 0.0939245,
		-0.15015967, -0.07292344, -0.15042665, 0.46306955,
		-0.00337431, -0.26442773, 0.20563623, -0.30521091,
		0.0522159, -0.48991753, -0.33204651, 0.04921531,
	}
	assert.True(t, noise.AlmostEqual(expected_noise, 0.0001))
	// Apply the filter
	filtered := GaussianFilter1D(noise, 2.0)
	expected_filtered := np.NpArray{
		0.09977654, 0.10837287, 0.11886541, 0.12445935,
		0.12476597, 0.12249207, 0.11452619, 0.09295791,
		0.05517371, 0.00632156, -0.04838469, -0.10741135,
		-0.16656225, -0.21293919, -0.23125834, -0.21803344,
		-0.18522694, -0.14769114, -0.1114015, -0.07799954,
		-0.05566714, -0.05537764, -0.07588471, -0.10085954,
		-0.11382477, -0.11091035, -0.09791311, -0.07987655,
		-0.0563267, -0.02617653, 0.00383111, 0.01905639,
		0.00902631, -0.02363039, -0.06869925, -0.11670316,
		-0.15785914, -0.18092825, -0.18276517, -0.17568301,
	}
	assert.True(t, filtered.AlmostEqual(expected_filtered, 0.0001))
}
