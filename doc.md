1.
我们知道了S_t, A_t, R_t+1, \hat{v}(S_t+1, w). 还有S_t+1.
现在，如果master接收了"A_t+1是什么"的query. 那么：
worker这边相当于是知道了\hat{v}(S_t+1, w).
所以，idea是，
知道\hat{v}(S_t+1, w)之后，先进行sample; 然后再进行update window的操作。
index t对应于如下的info.:
S_t, R_t+1, A_t, \hat{v}(S_t+1, w).

为什么是R_t+1? 答：请consider S_0时，这个时候R_0是没有意义的。
