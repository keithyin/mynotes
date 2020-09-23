#pragma once
#include <vector>

#include "ds_code_utils/print_utils.h"

class QuickSort {
   public:
    void Sort(std::vector<int> &arr) { partition(arr, 0, arr.size() - 1); }

    void partition(std::vector<int> &arr, int begin, int end) {
        if (begin >= end) {
            return;
        }
        int anchor = arr[begin];
        int i = begin;
        int j = end;

        while (i < j) {
            while (i < j && arr[j] >= anchor) j--;
            if (i < j) {
                arr[i++] = arr[j];
            }
            while (i < j && arr[i] <= anchor) i++;
            if (i < j) {
                arr[j--] = arr[i];
            }
        }
        arr[i] = anchor;

        partition(arr, begin, i - 1);
        partition(arr, i + 1, end);
    }
};

inline void TestQuichSort() {
    std::vector<int> arr = {2, 4, 1, 2, 3, 5, 6, 3, 9, 12, 8};
    QuickSort sorter;
    sorter.Sort(arr);
    print_utils::PrintVector(arr);
}