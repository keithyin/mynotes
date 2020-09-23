#pragma once

#include <iostream>
#include <vector>

#include "ds_code_utils/print_utils.h"

class MergeSort {
   public:
    void Sort(std::vector<int> &arr) {
        helper_vec_.resize(arr.size());
        merge(arr, 0, arr.size() - 1);
    }

    void merge(std::vector<int> &arr, int begin, int end) {
        if (begin >= end) {
            return;
        }
        int mid = (begin + end) / 2;
        merge(arr, begin, mid);
        merge(arr, mid + 1, end);

        int i = begin;
        int j = mid + 1;
        int cursor = begin;
        for (; i <= mid && j <= end;) {
            if (arr[i] < arr[j]) {
                helper_vec_[cursor++] = arr[i++];
            } else {
                helper_vec_[cursor++] = arr[j++];
            }
            std::cout << "arr.size()=" << arr.size() << " cursor=" << cursor << " begin=" << begin << " end=" << end
                      << " i=" << i << " j=" << j << " mid=" << mid << std::endl;
        }
        for (; j <= end;) {
            helper_vec_[cursor++] = arr[j++];
        }
        for (; i <= mid;) {
            helper_vec_[cursor++] = arr[i++];
        }
        for (int i = begin; i <= end; i++) {
            arr[i] = helper_vec_[i];
        }
    }

   private:
    std::vector<int> helper_vec_;
};

void TestMergeSort() {
    std::vector<int> arr = {2, 4, 1, 2, 3, 5, 6, 3, 9, 12, 8};
    MergeSort mergesort;
    mergesort.Sort(arr);
    print_utils::PrintVector(arr);
}