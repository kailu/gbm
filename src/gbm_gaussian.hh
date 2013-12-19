#ifndef GBM_GAUSSIAN_HH
#define GBM_GAUSSIAN_HH

#include "gbm_scalar_distribution.hh"

namespace gbm {

  class gaussian : public scalar_distribution {

    virtual double initial_f(const training_data& dat) const {
      const training_data::vector_type& ys = dat.get_y();
      const training_data::vector_type& offsets = dat.get_offset();
      const training_data::vector_type& weights = dat.get_weight();
      double sum=0, weight=0;
      for (R_len_t ind=0;ind!=dat.num_train();++ind) {
	sum += weights[ind] * (ys[ind] - offsets[ind]);
	weight += weights[ind];
      }
      return sum / weight;
    }

    virtual vector_type working_response(const training_data& dat,
					 const vector_type& F) const = 0;
    
  };

}


#endif
