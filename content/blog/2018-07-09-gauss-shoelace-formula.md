---
title: Gauss' shoelace formula
date: 2018-07-09 09:00:00
meta: true
math: true
---

There's a useful trick I learned the other day while working some geospatial
data. I needed to compute the areas of millions of polygons very quickly - which
initially looked like a fairly daunting task. It turns out that there's a very
fast and efficient method developed by Gauss that was designed for computing the
area of any simple polygon (a *simple* polygon is a closed-path flat shape
consisting of non-intersecting line segments - i.e. no curves and no overlaps).

<!-- more -->

It's called the *shoelace formula* (for reasons that will become clear later),
and it's quite elegant. Given a list of the vertices, the shoelace algorithm
breaks a polygon down into triangles and computes each of their areas instead.
Once these areas are known, they can all just be added together - which gives
you the area of the original polygon. What I find interesting is *just how
simple the algorithm is*.

Triangles have some handy properties; firstly, their areas can be computed
cheaply with a very simple closed-form solution - and secondly, any polygon can
be broken down into a number of triangles. This makes sense - mathematicians
like to solve difficult problems by breaking them down into smaller, easier
problems - and solving those instead.

---

Here's how it works. For the sake of simplicity, let's start with a
common-or-garden right-angled triangle.

![A simple right-angled triangle](/images/right-angled-triangle.png)

The area of this triangle is easily computed; $A = \frac{1}{2bh}$. Gauss'
shoelace formula looks a *little* more complicated...

$$
A = \frac{1}{2} \| \sum_{i=1}^{n-1} x_{i}y_{i+1} + x_{n}y_{1} - \sum_{i=1}^{n-1}
x_{i+1}y_{i} - x_{1}y_{n} \|
$$

where $(x_{i}, y_{i})$ are the coordinates of each vertex (thus $\mathbf{x}$ and
$\mathbf{y}$ are vectors containing the polygon's vertices). It isn't as
intimidating as it looks, and the algorithm is probably better explained using
an image (which I purloined sneakily from Wikipedia).

![By Job Bouwman - Own work, CC BY-SA 4.0](/images/shoelace.png)

Essentially, you cross-multiply each $x_{i}$ coordinate with each $y_{i-1}$
coordinate, and then divide the result by 2. As you work down through the vector
of vertices, this will have the effect of computing the area of each individual
triangle that comprises the polygon. Using our right-angled triangle from above,
here are three imaginary points - one corresponding to each vertex.

| Vertex | $\mathbf{x}$ | $\mathbf{y}$ |
|--------|--------------|--------------|
| a      | 3            | 5            |
| b      | 3            | 2            |
| c      | 8            | 2            |

Using Gauss' formula, area of this triangle is computed like so;

* $(3 \times 2) - (5 \times 3) = -9$
* $(3 \times 2) - (2 \times 8) = -10$
* $(8 \times 5) - (2 \times 3) = 34$ (*note here that we're "wrapping around" the top
  of the vector*)

Adding all of these areas together gives and dividing by two results in an area
of $7.5$. We can verify this quite easily: $\frac{1}{2} (5 \times 3) = 7.5$.
This is a bit over-complicated for a triangle - but this method works just as
well on polygons with an arbitrary number of vertices.

You may have noticed that one of the areas above is negative. The reason is that
the quantity $(x_{n-1}y_{n}) - (x_{n}y_{n-1})$ simply describes the rectangle
that has one particular vertex at its *top right*, and another at its *bottom
left*. For the right-angled triangle shown above, the area given by
$(x_{n-1}y_{n}) - (x_{n}y_{n-1})$ defines the rectangle with vertex $b$ at its
lower left, and vertex $d$ at its upper right. Dividing this area by two gives
the area of triangle $abc$.

So what about a more complex polygon? The same procedure applies, but the shape
gets broken up into right-angled triangles first.

![A quadrilateral broken up into right-angled
triangles](/images/complex-polygon.png)

---

So there we have it - Gauss' shoelace algorithm. It's so compact and elegant for
such a neat piece of geometric reasoning. When implementing this in a
programming language that supports vectorisation, it boils down to just four
lines of code...

```python
# See https://stackoverflow.com/a/49129646/2321244
def shoelace(xs, ys):
    main_area = np.dot(xs[:-1], ys[1:]) - np.dot(ys[:-1], xs[1:])
    correction = (xs[-1] * ys[0]) - (ys[-1] * xs[0])
    return np.abs(main_area + correction) / 2
```

...and runs in mere microseconds. Fantastic really, considering that Gauss died
over 150 years ago.