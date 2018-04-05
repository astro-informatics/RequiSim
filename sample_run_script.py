#__________________________________#
#      RequiSim                    #
#      Peter Taylor                #
#      Mullard Space Laboratory    #
#      University College London   #
#      2018                        #
#__________________________________#



from RequiSim import P_VWPO
from RequiSim import plot_k
from RequiSim import get_k_cell_boundaries
from RequiSim import get_z_cell_boundaries
import numpy as np


# load in the example knowledge matrix #
test = np.loadtxt('sample_knowledge_matrix.txt')
# compute and print the variance weighted overlap#
print "The variance weighted overlap for this knowledge matrix is:"
print P_VWPO(test)
#plot the diagonal of knowledge matrix#
plot_k(test)
#print the cell boundaries
print "The z cell boundaries are:"
print get_z_cell_boundaries()
print "The k [h Mpc ^ {-1}] cell boundaries are:"
print get_k_cell_boundaries()

