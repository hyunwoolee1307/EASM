#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# %%
x = np.linspace(0, 1)
y = np.linspace(0, 1)

# %%
fig = plt.figure(figsize=(6, 4), layout="constrained")
ax = plt.axes()
im = ax.plot(x, y)
plt.show()
plt.close()
# %%
