-1.

Dh(x) = Dg(f(x))Df(x), D等是Jacobian.

对于我们的问题，ACER中的trust region. 可以如下考虑：

\tfract{\partial\phi_\theta(x)}{\partial\theta}z^*

z^* 的前面部分是Jacobian. 上式的本质就是matrix之间的乘法。我们的idea是

将其写成single cost. 这样可以方便tf对其进行处理。

对z^* 进行stop_gradient. single cost可以写为：

\phi_\theta(x)^T z^* 

z^* 是一个column vector. 上式的idea是dot product, 所以，乘完之后会有sum的操作。

我们的问题中，\phi_\theta(x)是一个vector, 对于单个的transition.

其的form是\pi(a|s)这种。

0.
The link is good. See:
https://media.nips.cc/Conferences/2016/Slides/6198-Slides.pdf

Please note all the links to papers in the pdf.

They are also very important.

1.
我们知道了S_t, A_t, R_t+1, \hat{v}(S_t+1, w). 还有S_t+1.
现在，如果master接收了"A_t+1是什么"的query. 那么：
worker这边相当于是知道了\hat{v}(S_t+1, w).
所以，idea是，
知道\hat{v}(S_t+1, w)之后，先进行sample; 然后再进行update window的操作。
index t对应于如下的info.:
S_t, R_t+1, A_t, \hat{v}(S_t+1, w).

为什么是R_t+1? 答：请consider S_0时，这个时候R_0是没有意义的。

2.
workers这边的window的indices需要和master那边的对应起来。
master那边reply A_t+1时，有如下的info.:
S_t+1, A_t+1, \hat{v}(S_t+1, w).
其惟一不知道的是R_t+2.
master暂时不能save S_t+1等。因为master需要等worker将该info. save之后，才能save.
Otherwise, worker返回的sample就不能和master中的对应起来。

3.
master这边有size为1的buffer. 各个transition, 先进buffer, 然后，再进sliding window.
“然后”的含义，S_t+1计算好之后，S_t进sliding window.

is_over时，S_t亦进sliding window. master不必关心index是否valid. 这样的工作都将交给workers.

4.
关于cyclic queue的问题
我们不能在queue中多加一个slot用来区分：full和empty.
因为这样的话，之后np进行sample时会非常不方便。
所以，我们用valid数组来表示各个component是不是valid.
index不要从-1开始，而是从0开始。
这样的话，即使index对应的不是valid都没有关系，因为有valid的数组给我们是否是valid的info.

5.
关于is_over之后的处理：
S_t, A_t, episode over
sliding window仍会slide WINDOW_SIZE次。
注意slide时，将valid数组都set一下。
另外，成为dummy的component, 相应的weight reset为1e-8.

新加入的sliding window的transition, prob.为1e-8. (default)



























