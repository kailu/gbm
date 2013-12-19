#ifndef GBM_DATA_HH
#define GBM_DATA_HH

#include <set>
#include <exception>

#include <Rcpp.h>

namespace gbm {


  namespace {

    /*
       because of the way that optional parameters are passed in,
       we need a bit of cleverness here.

       for simplicity we just expand the missing values up to a zero
       vector of appropriate size, with downstream code
       simplifications.
    */
    Rcpp::NumericVector fix_na_value(const Rcpp::NumericVector& match,
				     const Rcpp::NumericVector& other) {
      if (other.size() == 1) {
	if (is_true(any(is_na(other)))) {
	  return Rcpp::NumericVector(match.size(), 0.0);
	}
      }
      return other;
    }


  class data {

  public:

    typedef Rcpp::NumericVector vector_type;
    typedef Rcpp::NumericMatrix matrix_type;

    data(const vector_type& y,
	 const matrix_type& predictors,
	 const vector_type& offset,
	 const vector_type& weight,
	 const vector_type& misc,
	 const R_len_t n_train) :
      y(y),
      offset(fix_na_value(y, offset)),
      weight(weight),
      misc(fix_na_value(y, misc)),
      predictors(predictors),
      n_train(n_train) {
      if (n_train > y.size()) {
	throw std::invalid_argument("more training cases than data?");
      }

      if (n_train == 0) {
	throw std::invalid_argument("you have no training cases.  fool.");
      }


      if (y.size() != predictors.nrow()) {
	throw std::invalid_argument("endpoint and predictors don't match");
      }

      if (y.size() != this->offset.size()) {
	throw std::invalid_argument("endpoint and offset don't match");
      }

      if (y.size() != weight.size()) {
	throw std::invalid_argument("endpoint and weight don't match");
      }

      if (y.size() != this->misc.size()) {
	throw std::invalid_argument("endpoint and misc don't match");
      }
    }

    const vector_type& get_y() const {return y;}
    const matrix_type& get_predictors() const {return predictors;}
    const vector_type& get_offset() const {return offset;}
    const vector_type& get_weight() const {return weight;}
    const vector_type& get_misc() const {return misc;}

    R_len_t num_predictors() const { return predictors.ncol(); }
    R_len_t num_cases() const { return predictors.nrow(); }
    R_len_t num_train() const { return n_train; }
    R_len_t num_groups() const {
      std::set<double> distinct(misc.begin(), misc.begin() + n_train);
      return distinct.size();
    }

  private:
    vector_type y, offset, weight, misc;
    matrix_type predictors;
    R_len_t n_train;
  };


  class training_data {
  public:

    typedef data::vector_type vector_type;
    typedef data::matrix_type matrix_type;

    training_data(const data& dat) : dat(dat) {}

    R_len_t num_predictors() const { return dat.num_predictors();}
    R_len_t num_cases() const { return dat.num_train(); }
    R_len_t num_train() const { return dat.num_train(); }
    R_len_t num_groups() const { return dat.num_groups(); }

    // these are leaky abstractions done badly

    const vector_type& get_y() const {return dat.get_y();}

    const vector_type& get_weight() const {return dat.get_weight();}
    const vector_type& get_offset() const {return dat.get_offset();}

    const vector_type& get_misc() const {return dat.get_misc();}

    const matrix_type& get_predictors() const {return dat.get_predictors();}

  private:
    const data& dat;
  };

}

#endif
