K-reciprocal Encoding
===

# Introduction
- This is my python implementation of a CVPR-2017 paper [Re-ranking Person Re-identification with k-reciprocal Encoding](https://arxiv.org/pdf/1701.08398.pdf)
- The authors have released their codes here in [github](https://github.com/zhunzhong07/person-re-ranking) too.

# How to run?
- `python3 main.py`
- Check `main.py` and replace `gallery_X`, `gallery_Y`, `query_X`, `query_X` with any data you want to try on.

# Performance
- The code is tested on a small portion of LFW dataset.
- mAP = 0.1553 using Euclidean Distance w Re-ranking
- mAP = 0.1025 using Euclidean Distance w/o Re-ranking

# Problems to resolved
- There are some parts in the original code from the authors which I do not understand. Those parts are not implemented in my code. Hence my implementation is a little but faster, but the performace is quite worse.
- mAP = 0.2182 using Euclidean Distance w the orginal code

## Problem 1
```python
original_dist = np.concatenate(
    [np.concatenate([q_q_dist, q_g_dist], axis=1),
    np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
axis=0)
```
- The probe and gallery are mixed here.
- One probe can be another probe's k-nearest neighbors.
- One probe can be another probe's k-reciprocal nearest neighbors.
- The overall computation of N, R and V are not independent within probes.

## Problem 2
```python
original_dist = np.power(original_dist, 2).astype(np.float32)
original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
```
- Why?

## Problem 3
```python
weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
```
- In the paper, the value is exp(-d) instead of whatever this is.


