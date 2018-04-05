#__________________________________#
#      RequiSim                    #
#      Peter Taylor                #
#      Mullard Space Laboratory    #
#      University College London   #
#      2018                        #
#__________________________________#

import numpy as np 
from matplotlib import pyplot as plt 
import math
from scipy import stats
from scipy.special import erf
import random

print "You are running RequiSim"

#__________________________________#
#  The resolution of the gird in   #
#   k and z for the power spectrum #
#  and resolution of pca_grid      #
#  Do Not Edit!                    #
#__________________________________#

resolution = 500
pca_num_k_bins = 15
pca_num_z_bins = 15
zoom_z_min = 0
zoom_z_max = 200
zoom_k_min = 200
zoom_k_max = 500
amp_change = 0.05




#__________________________________#
#  load the knowledge matrix       #
#   provided by the user           #
#                                  #
#__________________________________#

def covariance_fnc(reduced_fisher, eig_vectors, l_cut, knowledge_matrix):
    shape = np.shape(reduced_fisher)[0]
    bias_matrix = knowledge_matrix
    #rotate to pca frame
    pca_bias_matrix = np.dot(eig_vectors.T, np.dot(bias_matrix, eig_vectors))
    return pca_bias_matrix[:shape,:shape]


#__________________________________#
#  draw samples from covariance    #
#   defined by knowledge matrix    #
#                                  #
#__________________________________#

def gauss_draw(mean, covariance, n_samples):
    return np.random.multivariate_normal(mean, covariance, n_samples).T





#__________________________________#
#  load fisher and get the PCs.    #
#   Also do dimensional reduction  #
#   i.e. only return               #
#   fisher and PCs that contain    # 
#   more than frac_captured_info   #
#  of the sum of the variance.     #
#  Default is 99%                  #
#__________________________________#  

def load_fisher(resolution, pca_num_k_bins, pca_num_z_bins, zoom_z_min, zoom_z_max, zoom_k_min, zoom_k_max, amp_change, frac_captured_info, l_cut):
    #set the number of pc that we want to compute
    num_pc = pca_num_z_bins * pca_num_k_bins
    fisher_matrix = np.loadtxt(l_cut + '/fisher_matrix.txt')
    fisher_matrix = fisher_matrix / (2 * amp_change)
    z_samps = np.loadtxt(l_cut + '/z_samps.txt')
    k_samps = np.loadtxt(l_cut + '/k_samps.txt')
    #devide through by h for normal units
    k_samps = k_samps / 0.67
    eig_values, eig_vectors = np.linalg.eig(fisher_matrix)
    idx = eig_values.argsort()
    idx = idx[::-1]
    eig_values_2 = eig_values[idx]
    eig_vectors_2 = eig_vectors[:,idx]
    component_list = [i for i in range(num_pc)]
    z_bin_size = (zoom_z_max - zoom_z_min) / pca_num_z_bins
    k_bin_size = (zoom_k_max - zoom_k_min) / pca_num_k_bins
    number_params = pca_num_k_bins * pca_num_z_bins
    for i in range(num_pc):
        component = np.zeros((resolution, resolution))
        for pca_bin_number in range(number_params):
            z_bin_number = (pca_bin_number ) // pca_num_z_bins
            k_bin_number = (pca_bin_number ) % pca_num_k_bins
            component[zoom_z_min + z_bin_number * z_bin_size: zoom_z_min + (z_bin_number + 1) * z_bin_size, zoom_k_min + k_bin_number * k_bin_size: zoom_k_min + (k_bin_number + 1) * k_bin_size ] = eig_vectors_2[pca_bin_number][i].real
        component_list[i] = component
    total_eig_sum = np.sum(eig_values_2)
    eig_sum, information_frac, i = 0., 0., 0
    while information_frac < frac_captured_info :
        eig_sum += eig_values_2[i].real
        information_frac = eig_sum / total_eig_sum
        i += 1
    fisher = np.dot(eig_vectors.T, np.dot(fisher_matrix, eig_vectors))
    reduced_fisher = fisher[:i,:i].real
    return reduced_fisher, eig_vectors, total_eig_sum.real









#__________________________________#
#  Compute the Variance Weighted   #
#  Overlap.                        #
#__________________________________#  


def P_VWPO(knowledge_matrix, l_cut = 3000, n_samples = 5000, frac_captured_info = 0.99):
    l_cut = str(l_cut)
    # load fisher to find the errors #
    reduced_fisher, eig_vectors, total_eig_sum = load_fisher(resolution, pca_num_k_bins, pca_num_z_bins, zoom_z_min, zoom_z_max, zoom_k_min, zoom_k_max, amp_change, frac_captured_info, l_cut)
    n_dimensions = np.shape(reduced_fisher)[0]
    # load the knowledge matrix #
    covariance = covariance_fnc(reduced_fisher, eig_vectors, l_cut, knowledge_matrix)
    mean_vector = np.zeros(np.shape(reduced_fisher)[0])
    mean = 0.
    # sample from the knowledge matrix #
    bias_vector = gauss_draw(mean_vector, covariance, n_samples)
    overlap = np.zeros(n_dimensions)
    # find overlap for each PC #
    for j in xrange(n_dimensions):
        overlap[j] = 1. / n_samples * np.sum(1. - erf( 1. / (2. * math.sqrt(2.)) * np.abs(bias_vector[j,:] * reduced_fisher[j,j] ** 0.5)))
    # find relative importance of each component #
    weights = reduced_fisher.diagonal() / reduced_fisher.diagonal().sum()
    # weight to find the variance weighted overlap #
    p_vwpo =   np.sum(overlap * weights) / weights.sum()
    return p_vwpo







#__________________________________#
#  Plot the diagonal of the        #
#  knowledge matrix to             #
#  visualize  the uncertainty      #
#  on different regions of the     # 
#  power spectrum. This does not   #
#  account for correlations        #
#  between regions that are        #
# given by the off diagonal        #
#__________________________________#  

  
def plot_k(knowledge_matrix, l_cut = 3000):
    
    ##load files for plotting##
    z_samps = np.loadtxt('3000/z_samps.txt')
    #devide through by h for normal units
    k_samps = np.loadtxt('3000/k_samps.txt') / 0.67


    
    #get k-matrix onto PCA grid#
    number_of_bins = pca_num_k_bins * pca_num_z_bins
    k_matrix_diag = np.zeros((resolution, resolution))
    k_diag = np.diagonal(knowledge_matrix)
    z_bin_size = resolution / pca_num_z_bins
    k_bin_size = resolution / pca_num_k_bins
    for pca_bin_number in range(number_of_bins):
        z_bin_number = (pca_bin_number ) // pca_num_z_bins
        k_bin_number = (pca_bin_number ) % pca_num_k_bins
        k_matrix_diag[zoom_z_min + z_bin_number * z_bin_size: zoom_z_min + (z_bin_number + 1) * z_bin_size, zoom_k_min + k_bin_number * k_bin_size: zoom_k_min + (k_bin_number + 1) * k_bin_size ] = k_diag[pca_bin_number] 

    #plot#
    title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} 
    axis_font = {'fontname':'Arial', 'size':'22'}
    plt.pcolormesh(z_samps[0:200], k_samps[200:500], np.transpose(k_matrix_diag[0:200,200:500]), cmap=plt.cm.Blues)
    plt.title ('K-matrix Diagonal' ,**title_font)
    plt.xlim([0, z_samps[200]])
    plt.ylim([k_samps[200], k_samps[499]])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=15)
    plt.yscale('log')
    plt.xlabel('Redshift $(z)$' ,**axis_font)
    plt.ylabel('Wave-number $(k$ $[ h$ $Mpc ^ {-1}])$ ' ,**axis_font)
    axes = plt.gca()
    axes.tick_params('both', width=2, length = 7)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axes.xaxis.labelpad = 0
    [i.set_linewidth(2.) for i in axes.spines.itervalues()]
    plt.savefig('knowledge_matrix.png')
    plt.close()

    return 0.



#__________________________________#
#  get the k cell boundaries       #
#  if you want to produce          #
#   your own knowledge matrix      #
#__________________________________#  

def get_k_cell_boundaries():
    k_samps = np.loadtxt('3000/k_samps.txt')
    #devide through by h for normal units
    k_samps = k_samps / 0.67
    k_bin_size = (500 - 200) / 15
    k_cell_boundaries = np.zeros(16)
    k_cell_boundaries[15] = k_samps[499]
    for i in range(15):
        k_cell_boundaries[i] = k_samps[k_bin_size * i + 200]
    return k_cell_boundaries



#__________________________________#
#  get the z cell boundaries       #
#  if you want to produce          #
#   your own knowledge matrix      #
#__________________________________#


def get_z_cell_boundaries():
    z_samps = np.loadtxt('3000/z_samps.txt')
    z_bin_size = (200 - 0) / 15
    z_cell_boundaries = np.zeros(16)
    z_cell_boundaries[15] = z_samps[199]
    for i in range(15):
        z_cell_boundaries[i] = z_samps[z_bin_size * i]
    return z_cell_boundaries


    