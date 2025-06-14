import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

# Load the Bayesian Network from the .xdsl file
bn = gum.loadBN("MyNetwork.xdsl")

# Print the network structure to the console
print(bn)

# Display the network graphically (works best in Jupyter/IPython)
gnb.showBN(bn, size="9")