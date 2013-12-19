#ifndef GBM_SCALAR_DISTRIBUTION_HH
#define GBM_SCALAR_DISTRIBUTION_HH

#include "gbm_distribution.hh"

namespace gbm {
  class scalar_distribution : public distribution {
    virtual vector_type new_output(const training_data& dat) const {
      return vector_type(dat.num_train());
    }
  };
}

#endif
