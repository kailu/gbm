#include <memory>
#include <string>
#include <Rcpp.h>

#include "gbm_distributions.hh"
#include "gbm_tree_builder.hh"
#include "gbm_data.hh"

using namespace Rcpp;

namespace {

  std::auto_ptr<gbm::distribution>
  get_distribution(const List& args) {
    const std::string distribution = as<std::string>(args["distribution"]);
    typedef std::auto_ptr<gbm::distribution> res_type;

    if (distribution == "gaussian") {
      return res_type(new gbm::gaussian());
    } else if (distribution == "bernoulli") {
      return res_type(new gbm::bernoulli());
    }

    throw std::invalid_argument("distribution not known");
  }

  SEXP true_entry (List args) {
    // let's build the data
    const gbm::data data(args["Y"], args["X"], args["Offset"],
			 args["weights"], args["Misc"],
			 as<R_len_t>(args["nTrain"]));
    // something to create trees
    const gbm::tree_builder builder(get_distribution(args),
				    as<unsigned int>(args["interaction.depth"]),
				    as<R_len_t>(args["n.minobsinnode"]),
				    as<double>(args["shrinkage"]),
				    as<double>(args["bag.fraction"]));
    return builder.build_all(gbm::training_data(data),
			     as<R_len_t>(args["n.trees"]));
  }


  // this function is just so emacs doesn't get confused with
  // its indentation

  SEXP entry(SEXP args) {
    BEGIN_RCPP
    RNGScope scope;
    return true_entry(args);
    END_RCPP
  }

  R_ExternalMethodDef eMethods[] = {
    {"cGBM", (DL_FUNC) &entry, -1},
    {NULL, NULL, 0},
  };
}


extern "C" {

  void
  R_init_gbm(DllInfo *info) {
    R_registerRoutines(info, NULL, NULL, NULL, eMethods);
  }

}
