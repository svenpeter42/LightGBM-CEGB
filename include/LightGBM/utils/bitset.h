#ifndef LIGHTGBM_UTILS_BITSET_H_
#define LIGHTGBM_UTILS_BITSET_H_

#include <LightGBM/utils/common.h>

#include <vector>
#include <bitset>
#include <string>
#include <sstream>

namespace LightGBM {

class Bitset {
public:
  Bitset() { data_.resize(1, 0); }

  Bitset(int n) {
    data_.resize(n, 0);
  }

  Bitset(const std::string& str) {
    std::string s = str;
    s = Common::Trim(s);
    auto n = s.size();
    data_.resize(n, 0);
    for (size_t i = 0; i < n; ++i)
      if (s[i] == '1') {
        data_[i] = 1;
      }
  }

  void Set(int i, int n) {
    data_[i] = n;
  }

  bool Get(int i) const {
    if (static_cast<size_t>(i) >= data_.size()) {
      return false;
    }
    return data_[i];
  }

  std::string toString() {
    std::stringstream str_buf;
    for (size_t i = 0; i < data_.size(); ++i) {
      if (data_[i]) { 
        str_buf << 1; 
      } 
      else {
        str_buf << 0; 
      }
    }
    return str_buf.str();
  }

private:
  std::vector<bool> data_;
};

}
#endif // LIGHTGBM_UTILS_BITSET_H_