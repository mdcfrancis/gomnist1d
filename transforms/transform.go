package transforms

import (
	"fmt"
	np "github.com/mdcfrancis/gonp"
	"github.com/pa-m/randomkit"
)

type Rules struct {
	Padding        []int   `json:"padding"`
	TemplateLen    int     `json:"template_len"`
	ScaleCoeff     float64 `json:"scale_coeff"`
	MaxTranslation int     `json:"max_translation"`
	CorrNoiseScale float64 `json:"corr_noise_scale"`
	IidNoiseScale  float64 `json:"iid_noise_scale"`
	ShearScale     float64 `json:"shear_scale"`
	ShuffleSeq     bool    `json:"shuffle_seq"`
	FinalSeqLength int     `json:"final_seq_length"`
	Rnd            *randomkit.RKState
}

func (rules Rules) String() string {
	return fmt.Sprintf("Rules{Padding: %v, TemplateLen: %v, ScaleCoeff: %v, MaxTranslation: %v, CorrNoiseScale: %v, IidNoiseScale: %v, ShearScale: %v, ShuffleSeq: %v, FinalSeqLength: %v}", rules.Padding, rules.TemplateLen, rules.ScaleCoeff, rules.MaxTranslation, rules.CorrNoiseScale, rules.IidNoiseScale, rules.ShearScale, rules.ShuffleSeq, rules.FinalSeqLength)
}

func NewRules() Rules {
	return Rules{}
}

func (rules Rules) Pad(x np.NpArray, low int, high int) np.NpArray {
	ret := make(np.NpArray, len(x))
	for i := 0; i < len(x); i++ {
		ret = append(ret, x[i])
	}
	p := low + int(rules.Rnd.Float64()*float64((high-low+1)))
	return append(x, np.Zeros(p)...)

}

/*
coeff = scale * (np.random.rand() - 0.5)
x - coeff * np.linspace(-0.5, 0.5, len(x)))
*/

// Shear applies a shear transformation to the input array, default scale is 10
func (rules Rules) Shear(x np.NpArray, scale float64) np.NpArray {
	coeff := scale * (rules.Rnd.Float64() - 0.5)
	return x.Sub(
		np.LinSpace(-0.5, 0.5, x.Shape()).MulFloat64(coeff))
}

// Translate applies a translation to the input array, default max_translation is 3
func (rules Rules) Translate(x np.NpArray, max_translation int) np.NpArray {
	if len(x) < max_translation {
		err := fmt.Errorf("max_translation must be less than the length of the input array %d %d", len(x), max_translation)
		panic(err)
	}
	k := np.RandChoice(rules.Rnd, max_translation)
	return append(x[len(x)-k:], x[:len(x)-k]...)
}

// Interpolate applies a linear interpolation to the input array
func (rules Rules) Interpolate(x np.NpArray, n int) np.NpArray {
	scale := np.LinSpace(0, 1, len(x))
	new_scale := np.LinSpace(0, 1, n)
	return x.LinearInterpolate(scale, new_scale)
}

// CorrNoiseLike generates correlated noise
func (rules Rules) CorrNoiseLike(x np.NpArray, scale float64) np.NpArray {
	ret := np.RandN(rules.Rnd, x.Shape()).MulFloat64(scale)
	sigma := 2.0

	return GaussianFilter1D(ret, sigma)
}

// IidNoiseLike generates iid noise
func (rules Rules) IidNoiseLike(x np.NpArray, scale float64) np.NpArray {
	return np.RandN(rules.Rnd, x.Shape()).MulFloat64(scale)
}

// Transform applies a series of transformations to the input array
func (rules Rules) Transform(x np.NpArray, y np.NpArray) (np.NpArray, np.NpArray) {
	new_x := rules.Pad(x.AddFloat64(1e-8), rules.Padding[0], rules.Padding[1])              // pad
	new_x = rules.Interpolate(new_x, rules.TemplateLen+rules.Padding[len(rules.Padding)-1]) // dilate
	new_y := rules.Interpolate(y, rules.TemplateLen+rules.Padding[len(rules.Padding)-1])
	factor := 1 + rules.ScaleCoeff*(rules.Rnd.Float64()-0.5)
	new_x = new_x.MulFloat64(factor)                     // scale
	new_x = rules.Translate(new_x, rules.MaxTranslation) // translate

	// add corrNoiseLike
	mask := new_x.Cond(func(x float64) float64 {
		if x != 0 {
			return 1
		}
		return 0
	})
	oneMinus := func(a np.NpArray) np.NpArray {
		ret := make(np.NpArray, len(a))
		for i := 0; i < len(a); i++ {
			ret[i] = 1 - a[i]
		}
		return ret
	}

	/*
		mask = new_x != 0
			    new_x = mask * new_x + (1 - mask) * corr_noise_like(new_x, args.corr_noise_scale)
			    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)

	*/
	corrNoiseLike := rules.CorrNoiseLike(new_x, rules.CorrNoiseScale)
	masked_noise := corrNoiseLike.Mul(oneMinus(mask))
	new_x = new_x.Mul(mask).Add(masked_noise)

	iidNoiseLike := rules.IidNoiseLike(new_x, rules.IidNoiseScale)
	new_x = new_x.Add(iidNoiseLike)

	new_x = rules.Shear(new_x, rules.ShearScale)

	new_x = rules.Interpolate(new_x, rules.FinalSeqLength) // subsample
	new_y = rules.Interpolate(new_y, rules.FinalSeqLength)
	return new_x, new_y
}
