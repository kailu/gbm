#ifndef GBM_TREE_BUILDER_HH
#define GBM_TREE_BUILDER_HH

#include <iterator>
#include <iostream>

#include <exception>
#include <memory>
#include <vector>

#include <Rcpp.h>

#include "gbm_data.hh"

namespace gbm {

  class tree_builder {
  public:
    tree_builder(std::auto_ptr<distribution> dist,
		 const unsigned int max_depth,
		 const R_len_t min_obs_in_node,
		 const double shrinkage,
		 const double bag_fraction) :
      dist(dist),
      max_depth(max_depth),
      min_obs_in_node(min_obs_in_node),
      shrinkage(shrinkage),
      bag_fraction(bag_fraction) {
      if ((bag_fraction <= 0) || (bag_fraction > 1)) {
	throw std::invalid_argument("bag fraction is insane");
      }
      if (shrinkage <= 0) {
	throw std::invalid_argument("shrinkage is insane");
      }
    }

    Rcpp::List build_all(const training_data& dat, R_len_t num_trees) const {
      const double initial_f = dist->initial_f(dat);
      typedef distribution::matrix_type matrix_type;
      Rcpp::NumericVector f = dist->new_output(dat);
      f.fill(initial_f);
      Rcpp::List res;
      Rcpp::NumericVector train_error(num_trees, 0);
      Rcpp::NumericVector valid_error(num_trees, 0);
      Rcpp::NumericVector oobag_improve(num_trees, 0);

      for (R_len_t ind=0; ind!=num_trees; ++ind) {
	distribution::bag_type bag = dist->get_bag(bag_fraction, dat);
      }

      res["initF"] = Rcpp::wrap(initial_f);
      res["train.error"] = train_error;
      res["valid.error"] = valid_error;
      res["oobag.improve"] = oobag_improve;
      res["bag.fraction"] = Rcpp::wrap(bag.fraction);
      res["interaction.depth"] = Rcpp::wrap(max_depth);
      res["n.minobsinnode"] = Rcpp::wrap(min_obs_in_node);
      res["shrinkage"] = Rcpp::wrap(shrinkage);
      res["nTrain"] = Rcpp::wrap(dat.num_train());
      return res;
    }

  private:
    double shrinkage, bag_fraction;
    unsigned int max_depth, min_obs_in_node;
    std::auto_ptr<distribution> dist;
  };

}

#endif
