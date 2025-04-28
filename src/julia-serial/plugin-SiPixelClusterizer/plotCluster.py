import sys
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt

x = [1, 2, 3, 1, 2, 3, 1, 2]
y = [1, 1, 1, 2, 2, 2, 3, 3]

plt.scatter(x, y)
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()

# we expect this to be 1 cluster only 