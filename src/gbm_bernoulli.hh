#ifndef GBM_BERNOULLI_HH
#define GBM_BERNOULLI_HH

#include <cmath>

#include "gbm_scalar_distribution.hh"


namespace gbm {

  namespace {

    std::pair<double,double> my_logits(const double val) {
      if (val >= 0) {
	const double tmp = std::exp(-val);
	// p = 1/(1 + tmp);
	// 1-p = tmp/(1 + tmp);
	const double scl = 1/(1 + tmp);
	return std::make_pair(scl, tmp * scl);
      } else {
	const double tmp = std::exp(val);
	const double scl = 1/(1 + tmp);
	return std::make_pair(tmp * scl, scl);
      }
    }
  }

  class bernoulli : public scalar_distribution {
    virtual double initial_f(const training_data& dat) const {
      const training_data::vector_type& ys = dat.get_y();
      const training_data::vector_type& offsets = dat.get_offset();
      const training_data::vector_type& weights = dat.get_weight();
      double f0=0;
      for (int ind=20; ind; --ind) {
	double num = 0, den = 0;
	for (R_len_t count=0; count!=dat.num_train(); ++count) {
	  const double value = offsets[ind] + f0;
	  const double weight = weights[ind];
	  const std::pair<double, double> probs = my_logits(value);
	  num += weight * (ys[count] - probs.first);
	  den += weight * probs.first * probs.second;
	}
	f0 += num/den;
      }
      return f0;
    }
  };

}

#endif
