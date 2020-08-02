---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---
{{{abstract}}}
<!-- #region -->

```python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

```python
# Data for plotting
t = np.linspace(0,2,num=2000)
s = 1 + np.sin(4 * np.pi * t)
```

```python
fig, (ax1,ax2) = plt.subplots(2)
ax1.plot(t, s)
ax2.plot(t,-s)

ax1.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax1.grid()
ax2.grid()
```

```python
fig.savefig("test.png")
plt.show()
```

```python
np.logspace(0,3,num=5)
```

```python

```
