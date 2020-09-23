#include <iostream>
#include <string>
#include <vector>

inline int maxValInVec(const std::vector<int>& values) {
    int maxVal = 0;
    for (const auto& val : values) {
        maxVal = maxVal < val ? val : maxVal;
    }
    return maxVal;
}

// dp[i,j]; i,j的最长公共子序列
inline int MaxCommonSubSeq(const std::string& str1, const std::string& str2) {
    std::vector<std::vector<int>> dp;
    for (int i = 0; i < str1.length() + 1; i++) {
        std::vector<int> initvec(str2.length() + 1, 0);
        dp.push_back(initvec);
    }
    for (int i = 0; i < str1.length(); i++) {
        for (int j = 0; j < str2.length(); j++) {
            int maxlen = maxValInVec({dp[i][j] + (str1[i] == str2[j]), dp[i][j + 1], dp[i + 1][j]});
            dp[i + 1][j + 1] = maxlen;
        }
    }
    return dp[str1.length()][str2.length()];
}

inline void MaxCommonSubSeqTestCase() {
    std::string a = "hello";
    std::string b = "ell";
    int len = MaxCommonSubSeq(a, b);
    std::cout << "MaxCommonSubSeq.res=" << len << std::endl;
}