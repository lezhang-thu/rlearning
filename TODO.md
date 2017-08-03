1.
master那里，需要维护一个window.
里面是states. size为500.

2.
worker那里，并不需要维护states的window.
但是有对应的prob. 然后，还有它需要master告诉它，除了之前的action之外，还有
\hat{v}的值。

3.
为了利用np.random.choice的方法，我们需要用np array.
所以，本质上来说，为了代码清晰，我们采取维护一个valid的array. 来表示prob.中相应的
index是不是对应于真实的state, 还是对应于dummy.

4.
先sample, 然后再save. Why? 因为这个时候知道了\hat{v}(S_t+1, w).
例如，知道了S_t, A_t, R_t+1, \hat{v}(S_t, w).
S_t+1, 问master, A_t+1是什么。

5.
tensorpack中，Evaluator实在是消耗资源了。在common.py中，是20个进程。
然后，每一个默认是评估50个episodes. 所以，整个的，太消耗了。

将其改成了最多5个进程。然后，评估是2个episodes.
所以，总的是10个epsiodes. 可以了。


