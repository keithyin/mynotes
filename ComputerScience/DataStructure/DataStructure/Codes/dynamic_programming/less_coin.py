

def LessCoin(target, coins):
    INF = 10000000000
    dp = [INF] * (target+1)
    dp[0] = 0
    for i in range(target):
        i += 1
        minCoinNum = INF
        for coin in coins:
            if (i - coin) >= 0:
                minCoinNum = min([dp[i-coin]+1, minCoinNum])
        dp[i] = minCoinNum
    return dp[target]


def NumberOfMethodThatGiveYouChange():
    pass


if __name__ == "__main__":
    target = 5
    coins = [6]
    print(LessCoin(target, coins))
