package transforms

import (
	"fmt"
	np "github.com/mdcfrancis/gonp"
	"github.com/pa-m/randomkit"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPadAddsCorrectPadding(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3}
	padded := rules.Pad(x, 1, 3)
	assert.GreaterOrEqual(t, len(padded), len(x)+1)
	assert.LessOrEqual(t, len(padded), len(x)+3)
}

func TestShearAppliesCorrectTransformation(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3}
	sheared := rules.Shear(x, 10)
	assert.NotEqual(t, x, sheared)
}

func TestRules_TranslateAppliesCorrectTranslation(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3, 4, 5}
	translated := rules.Translate(x, 3)
	assert.NotEqual(t, x, translated)
}

func TestRules_TranslatePanicsIfMaxTranslationIsGreaterThanLength(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3}
	assert.Panics(t, func() { rules.Translate(x, 4) })
}

func TestRules_InterpolateGeneratesCorrectLength(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3}
	interpolated := rules.Interpolate(x, 5)
	assert.Equal(t, 5, len(interpolated))
}

func TestRules_IidNoiseLikeGeneratesNoise(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3}
	noise := rules.IidNoiseLike(x, 1)
	assert.Equal(t, len(x), len(noise))
}

func TestRules_TransformAppliesAllTransformations(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	rules.Padding = []int{1, 10}
	rules.TemplateLen = 5
	rules.ScaleCoeff = 0.5
	rules.MaxTranslation = 2
	rules.CorrNoiseScale = 0.1
	rules.IidNoiseScale = 0.1
	rules.ShearScale = 0.1
	rules.FinalSeqLength = 16
	x := np.NpArray{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	y := np.NpArray{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	newX, newY := rules.Transform(x, y)
	assert.Equal(t, 16, len(newX))
	assert.Equal(t, 16, len(newY))
}

func TestRules_IidNoiseLike(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{1, 2, 3, 4}
	noise := rules.IidNoiseLike(x, 2e-2)
	fmt.Println(noise)

	expect := np.NpArray{0.00993428, -0.00276529, 0.01295377, 0.0304606}
	assert.Equal(t, len(x), len(noise))
	for i, v := range noise {
		assert.InDelta(t, expect[i], v, 0.0001)
	}
}

func TestRules_Shear(t *testing.T) {
	rules := NewRules()
	rules.Rnd = randomkit.NewRandomkitSource(42)
	x := np.NpArray{0.0, 0.23693955, 0.35540933, 0.41464421, 0.4738791,
		0.4738791, 0.4738791, 0.4738791, 0.41464421, 0.35540933, 0.23693955, 0.0}
	expect := np.NpArray{-0.04704746, 0.19844618, 0.32547004, 0.39325901,
		0.46104798, 0.46960206, 0.47815614, 0.48671023, 0.43602942, 0.38534862,
		0.27543292, 0.04704746}

	for i, v := range rules.Shear(x, 0.75) {
		assert.InDelta(t, expect[i], v, 0.0001)
	}
}

func ExampleRules_String() {
	rules := NewRules()
	rules.Padding = []int{1, 10}
	rules.TemplateLen = 5
	rules.ScaleCoeff = 0.5
	rules.MaxTranslation = 2
	rules.CorrNoiseScale = 0.1
	rules.IidNoiseScale = 0.1
	rules.ShearScale = 0.1
	rules.FinalSeqLength = 16
	fmt.Println(rules.String())
	// Output:
	// Rules{Padding: [1 10], TemplateLen: 5, ScaleCoeff: 0.5, MaxTranslation: 2, CorrNoiseScale: 0.1, IidNoiseScale: 0.1, ShearScale: 0.1, ShuffleSeq: false, FinalSeqLength: 16}
}
