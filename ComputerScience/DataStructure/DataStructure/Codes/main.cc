#include "dynamic_programming/common_seq.h"
#include "sort/heap_sort.h"
#include "sort/merge_sort.h"
#include "sort/quick_sort.h"
int main() {
    TestMergeSort();
    TestQuichSort();
    TestHeapSort();
    MaxCommonSubSeqTestCase();
}
