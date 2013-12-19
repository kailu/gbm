#ifndef GBM_DISTRIBUTION_HH
#define GBM_DISTRIBUTION_HH

#include <vector>

#include <Rcpp.h>

#include "gbm_data.hh"

namespace gbm {

  class distribution {
  public:
    typedef std::vector<R_len_t> bag_type;
    typedef Rcpp::NumericMatrix matrix_type;
    typedef Rcpp::NumericVector vector_type;

    distribution() {}
    virtual ~distribution() {}

    virtual bag_type get_bag(const double bag_fraction,
			     const training_data& dat) const {
      const R_len_t num_train = dat.num_train();
      const R_len_t bag_size = bag_fraction * num_train;

      if ((bag_size <= 0) || (bag_size > num_train)) {
	throw std::invalid_argument("insane bag");
      }

      bag_type res;
      for (R_len_t ind = 0; ind != num_train; ++ind) {
	if (unif_rand() * (num_train - ind) < (bag_size - res.size())) {
	  res.push_back(ind);
	}
      }

      return res;

    }

    virtual double initial_f(const training_data& dat) const = 0;
    virtual vector_type working_response(const training_data& dat,
					 const vector_type& F) const = 0;
    virtual vector_type new_output(const training_data& dat) const = 0;

  };

}

#endif
