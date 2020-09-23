#pragma once

#include <vector>

#include "ds_code_utils/print_utils.h"

class Heap {
   public:
    void Insert(int value) {
        values_.push_back(value);
        swim(values_.size() - 1);
    }
    int Del() {
        if (values_.size() < 1) {
            throw "heap is empth";
        }
        int max_val = values_[1];
        swap(1, values_.size() - 1);
        values_.resize(values_.size() - 1);
        sink(1);
        return max_val;
    }

   private:
    void sink(int pos) {
        while (2 * pos < values_.size()) {
            int max_child_pos = 2 * pos;
            if (2 * pos + 1 < values_.size() && values_[max_child_pos] < values_[max_child_pos + 1]) {
                max_child_pos++;
            }
            if (values_[pos] > values_[max_child_pos]) break;
            swap(pos, max_child_pos);
            pos = max_child_pos;
        }
    }
    void swim(int pos) {
        while (pos > 1) {
            int father = pos / 2;
            if (values_[father] < values_[pos]) {
                swap(father, pos);
                pos = father;
            } else {
                break;
            }
        }
    }

    void swap(int i, int j) {
        int tmp = values_[i];
        values_[i] = values_[j];
        values_[j] = tmp;
    }

   private:
    std::vector<int> values_ = {0};
};

class HeapSort {
   public:
    void Sort(std::vector<int>& values) {
        for (auto& v : values) {
            heap_.Insert(v);
        }
        for (int i = 0; i < values.size(); i++) {
            values[i] = heap_.Del();
        }
    }

   private:
    Heap heap_;
};

void TestHeapSort() {
    std::vector<int> arr = {2, 4, 1, 2, 3, 5, 6, 3, 9, 12, 8};
    HeapSort sort;
    sort.Sort(arr);
    print_utils::PrintVector(arr);
}