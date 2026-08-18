---
title: "Typography Specimen"
date: 2026-08-18
description: "A sample essay exercising every element this blog can render. Delete once real posts exist."
---

This post exists to exercise every element an essay here can contain.
It doubles as a writing template — copy its frontmatter, delete it when
real posts exist.

## Prose and emphasis

Body text sits in a narrow column at a comfortable line height. Inline
elements include *emphasis*, **strong emphasis**, `inline code`, and
[links](https://example.com). Quotations render like this:

> The purpose of computing is insight, not numbers.
> — Richard Hamming

## Mathematics

Inline math flows with the text: the loss $\mathcal{L}(\theta) = -\sum_i y_i \log \hat{y}_i$
should sit on the baseline. Display math gets its own block:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

## Code

```python
def attention(q, k, v):
    scores = q @ k.T / math.sqrt(k.shape[-1])
    return softmax(scores) @ v
```

## Figures

<figure>
  <img src="/images/specimen-figure.svg" alt="A sample line chart" />
  <figcaption>Fig. 1. Figures carry numbered captions in muted text.</figcaption>
</figure>

## Lists

1. Ordered lists for sequences.
2. Second item.

- Unordered lists for everything else.
- Second item.
