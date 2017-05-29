#include <LightGBM/boosting.h>
#include "gbdt.h"
#include "dart.hpp"
#include "goss.hpp"
#include "cegb.hpp"

namespace LightGBM {

std::string GetBoostingTypeFromModelFile(const char* filename) {
  TextReader<size_t> model_reader(filename, true);
  std::string type = model_reader.first_line();
  return type;
}

bool Boosting::LoadFileToBoosting(Boosting* boosting, const char* filename) {
  if (boosting != nullptr) {
    TextReader<size_t> model_reader(filename, true);
    model_reader.ReadAllLines();
    std::stringstream str_buf;
    for (auto& line : model_reader.Lines()) {
      str_buf << line << '\n';
    }
    if (!boosting->LoadModelFromString(str_buf.str()))
      return false;
  }

  return true;
}

Boosting* Boosting::CreateBoosting(const std::string& type, const char* filename) {
  if (filename == nullptr || filename[0] == '\0') {
    if (type == std::string("gbdt")) {
      return new GBDT();
    } else if (type == std::string("dart")) {
      return new DART();
    } else if (type == std::string("cegb")) {
      return new CEGB();
    } else if (type == std::string("goss")) {
      return new GOSS();
    } else {
      return nullptr;
    }
  } else {
    std::unique_ptr<Boosting> ret;
    auto type_in_file = GetBoostingTypeFromModelFile(filename);
    if (type_in_file == std::string("cegb_tree") && type == std::string("cegb")) {
      ret.reset(new CEGB());
    } else {
      Log::Fatal("type_in_file and type mismatch: %s and %s", type_in_file.c_str(), type.c_str());
    }

    if (type_in_file == std::string("tree")) {
      if (type == std::string("gbdt")) {
        ret.reset(new GBDT());
      } else if (type == std::string("dart")) {
        ret.reset(new DART());
      } else if (type == std::string("cegb")) {
        ret.reset(new CEGB());
      } else if (type == std::string("goss")) {
        ret.reset(new GOSS());
      } else {
        Log::Fatal("unknown boosting type %s", type.c_str());
      }
      LoadFileToBoosting(ret.get(), filename);
    } else {
      Log::Fatal("unknown submodel type in model file %s", filename);
    }
    return ret.release();
  }
}

Boosting* Boosting::CreateBoosting(const char* filename) {
  auto type = GetBoostingTypeFromModelFile(filename);
  std::unique_ptr<Boosting> ret;
  if (type == std::string("tree")) {
    ret.reset(new GBDT());
  } else if (type == std::string("cegb_tree")) {
    ret.reset(new CEGB());
  } else {
    Log::Fatal("unknown submodel type in model file %s", filename);
  }
  LoadFileToBoosting(ret.get(), filename);
  return ret.release();
}



void Boosting::PredictCost(
    const double* features, double* output) const
{
  Log::Fatal("Boosting::PredictCost has not been implemented yet.");
}


void Boosting::PredictMulti(
    const double* features, double* output_raw, double* output,
    double* leaf, double *cost) const
{
  Log::Fatal("Boosting::PredictMulti has not been implemented yet.");
}

}  // namespace LightGBM
