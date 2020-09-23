#pragma once

#include <iostream>
#include <vector>
namespace print_utils {
template <typename T>
inline void PrintVector(const std::vector<T>& arr) {
    for (auto& v : arr) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}
}  // namespace print_utils
