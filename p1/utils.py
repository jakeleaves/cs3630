import gtsam
import math
import numpy as np

from enum import Enum
from gtbook.discrete import Variables


### ENUMS ###
class Trash(Enum):
    CARDBOARD = 0
    PAPER = 1
    CAN = 2
    SCRAP_METAL = 3
    BOTTLE = 4


class Bin(Enum):
    GLASS_BIN = 0
    METAL_BIN = 1
    PAPER_BIN = 2
    NOP = 3


class Detection(Enum):
    BOTTLE = 0
    CARDBOARD = 1
    PAPER = 2


### CONSTANTS ###
# All possible trash categories
CATEGORIES = ['cardboard', 'paper', 'can', 'scrap_metal', 'bottle']

# All possible actions/bins (nop means no action)
ACTIONS = ['glass_bin', 'metal_bin', 'paper_bin', 'nop']

# TODO 4:
'''
    Fill out the cost table with corresponding costs, where the rows correspond
    to ACTIONS and the columns correspond to CATEGORIES.
'''
###############################################################################
#                             START OF YOUR CODE                              #
###############################################################################
COST_TABLE = np.array([[2, 2, 4, 6, 0],
                       [1, 1, 0, 0, 2],
                       [0, 0, 5, 10, 3],
                       [1, 1, 1, 1, 1]])
###############################################################################
#                              END OF YOUR CODE                               #
###############################################################################

### PROBABILITY ###
# store probability values
class Probability:
    def __init__(self):
        self.variables = Variables()
        self.categories = CATEGORIES
        self.Category = self.variables.discrete('Category', self.categories)
        self.Conductivity = self.variables.binary('Conductivity')
        self.Detection = self.variables.discrete('Detection', ['bottle', 'cardboard', 'paper'])

    # TODO 1:
    # Prior probabilities
    def get_category_prior(self):
        '''
        Returns the prior probabilities of the trash categories.

            Parameters:
                None

            Returns:
                category_prior (gtsam.DiscreteDistribution): a DiscreteDistribution
                    that summarizes the prior probabilities of all trash categories
        '''
        category_prior = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        Category = self.variables.discrete("Category", self.categories)
        category_prior = gtsam.DiscreteDistribution(Category, "200/300/250/200/50")

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return category_prior

    # TODO 2:
    # Prior probabilities PMF
    def get_category_prior_pmf(self):
        '''
        Returns the probability mass function (PMF) of the prior probabilities
        of the trash categories.

            Parameters:
                None

            Returns:
                category_prior_pmf (list): a list of the PMF
        '''
        category_prior_pmf = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        category_prior_pmf = [0.2, 0.3, 0.25, 0.2, 0.05]
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return category_prior_pmf

    # TODO 5:
    # 1. Conductivity - binary sensor
    def get_pCT(self):
        '''
        Returns P(Conductivity | Trash Category)

            Parameters:
                None

            Returns:
                pCT (gtsam.DiscreteConditional): a DiscreteConditional that
                    indicates the conditinal probability of conductivity given
                    the trash category
        '''
        pCT = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        pCT = gtsam.DiscreteConditional(self.Conductivity, [self.Category], "99/1 99/1 10/90 15/85 95/5")
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return pCT

    # TODO 6:
    # 2. Detection - multi-valued sensor
    def get_pDT(self):
        '''
        Returns P(Detection | Trash Category)

            Parameters:
                None

            Returns:
                pDT (gtsam.DiscreteConditional): a DiscreteConditional that
                    indicates the conditinal probability of camera detection
                    given the trash category
        '''
        pDT = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        pDT = gtsam.DiscreteConditional(self.Detection, [self.Category], "2/88/10 2/20/78 33/33/34 33/33/34 95/2/3")

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return pDT

    # TODO 7:
    # 3. Weight - continuous-valued sensor
    def get_pWT(self):
        '''
        Returns P(Weight | Trash Category)

            Parameters:
                None

            Returns:
                pWT (np.array): a numpy array of lists that consists of the means
                    and standard deviations that define the weight distribution of each
                    trash category

        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        pWT = np.array([[20, 10], [5, 5], [15, 5], [150, 100], [300, 200]])
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return pWT


### HELPER FUNCTIONS ###
def Gaussian(x, mu=0.0, sigma=1.0):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2)

# TODO 17
def fit_log_normal(data):
    '''
    Returns mu, sigma for a log-normal distribution

        Parameters:
            data (list of floats): A list of positive floats that represent the weight of an item

        Returns:
            mu (float), sigma (float): The mu and sigma for a log-normal distribution
    '''
    mu = None
    sigma = None
    ###############################################################################
    #                             START OF YOUR CODE                              #
    ###############################################################################

    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return mu, sigma