import rclpy
import gtsam
import message_filters

import numpy as np

from enum import Enum
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int16

import utils
from utils import Bin, COST_TABLE, Detection, Gaussian, Probability, Trash

np.random.seed(3630)

class PerceptionOption(Enum):
    LIKELIHOOD_NO_SENSORS = 0
    LIKELIHOOD_GIVEN_WEIGHT = 1
    LIKELIHOOD_GIVEN_DETECTION = 2
    BAYES_GIVEN_WEIGHT = 3
    BAYES_GIVEN_THREE_SENSORS = 4


class Robot(Node):
    '''
    This class simulates the robot making a sorting decision

        PARAMETERS:
            perception_option (PerceptionOption): Which perception mode to use when making a decision
            use_ros (bool): Whether to setup the ros infrastructure (set to false for unittests)
    '''

    def __init__(self, perception_option, use_ros=True):
        if use_ros:
            super().__init__('robot')

        self.P = Probability()
        self.perception_option = perception_option
        if use_ros:
            self.ros_init()

    def ros_init(self):
        # Create subscribers
        self.iteration_subscriber = message_filters.Subscriber(self, Int16, 'iteration')
        self.conductivity_subscriber = message_filters.Subscriber(self, Bool, 'conductivity')
        self.detection_subscriber = message_filters.Subscriber(self, Int16, 'detection')
        self.weight_subscriber = message_filters.Subscriber(self, Float32, 'weight')
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.iteration_subscriber, self.conductivity_subscriber, self.detection_subscriber,
             self.weight_subscriber],
            1,
            slop=0.5,
            allow_headerless=True
        )
        self.time_synchronizer.registerCallback(self.subscriber_callback)

        self.decision_publisher = self.create_publisher(Int16, 'decision', 1)
        self.get_logger().info('decision_publisher has been started.')

    def subscriber_callback(self, iteration_msg, conductivity_msg, detection_msg, weight_msg):
        iteration = iteration_msg.data
        conductivity = conductivity_msg.data
        detection = detection_msg.data
        weight = weight_msg.data

        # log
        self.get_logger().info(
            f'Received sample {iteration}:' +
            f'\tConductivity: {conductivity};' +
            f'\tDetection: {Detection(detection).name};' +
            f'\tWeight: {weight} grams;'
        )

        ### PERCEPTION ###
        probabilities = None
        if self.perception_option == PerceptionOption.LIKELIHOOD_NO_SENSORS:
            probabilities = self.likelihood_no_sensors()
        elif self.perception_option == PerceptionOption.LIKELIHOOD_GIVEN_WEIGHT:
            probabilities = self.likelihood_given_weight(weight)
        elif self.perception_option == PerceptionOption.LIKELIHOOD_GIVEN_DETECTION:
            probabilities = self.likelihood_given_detection(detection)
        elif self.perception_option == PerceptionOption.BAYES_GIVEN_WEIGHT:
            probabilities = self.bayes_given_weight(weight)
        elif self.perception_option == PerceptionOption.BAYES_GIVEN_THREE_SENSORS:
            probabilities = self.bayes_given_three_sensors(conductivity, detection, weight)
        else:
            raise Exception("Unknown perception option")
        self.get_logger().info(f'The robot believes the trash is {self.perceive(probabilities)}')

        ### DECISION ###
        action = self.make_decision(probabilities)
        self.get_logger().info(f'The robot has decided to put the trash in the {Bin(action).name}')

        # Publish decision
        decision_msg = Int16()
        decision_msg.data = int(action)
        self.decision_publisher.publish(decision_msg)

    ### PERCEPTION ###
    def perceive(self, likelihoods):
        return Trash(likelihoods.index(max(likelihoods))).name

    # TODO 11:
    def likelihood_no_sensors(self):
        '''
        Returns the likelihoods of all trash categories using only priors,
        aka no sensors.

            Parameters:
                None

            Returns:
                likelihoods (list): a list of likelihoods of each trash category
        '''
        likelihoods = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        likelihoods = self.P.get_category_prior_pmf()
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return likelihoods

    # TODO 12:
    def likelihood_given_weight(self, weight):
        '''
        Returns the likelihoods of all trash categories using only the weight
        sensor (no priors)

            Parameters:
                weight (double): a double indicating the weight of trash

            Returns:
                likelihoods (list): a list of likelihoods of each trash category
        '''
        likelihoods = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        likelihoods = np.array([utils.Gaussian(weight, *(self.P.get_pWT()[index])) for index in range(5)])
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return likelihoods

    # TODO 13:
    def likelihood_given_detection(self, detection):
        '''
        Returns the likelihoods of all trash categories using only the detection
        sensor (no priors)

            Parameters:
                detection (int): an int indicating the sampled detection, the
                    int-detection mapping is available in utils.py

            Returns:
                likelihoods (list): a list of likelihoods of each trash category
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        likelihoods = [v for (k, v) in (self.P.get_pDT().likelihood(detection)).enumerate()]
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return likelihoods

    # TODO 14:
    def bayes_given_weight(self, weight):
        '''
        Returns the posteriors of all trash categories by combining the weight
        sensor and the priors

            Parameters:
                weight (double): a double indicating the weight of the trash

            Returns:
                posteriors (list): a list of posterior probabilities of each trash category
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        category_prior = gtsam.DiscreteDistribution(self.P.Category, "200/300/250/200/50")
        weight_factor = gtsam.DecisionTreeFactor(self.P.Category, self.likelihood_given_weight(weight))

        posteriors = gtsam.DiscreteDistribution(weight_factor * category_prior).pmf()

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return posteriors

    # TODO 15:
    # Bayes with three sensors
    def bayes_given_three_sensors(self, conductivity, detection, weight):
        '''
        Returns the posteriors of all trash categories by combining all three
        sensors and the priors

            Parameters:
                conductivity (int): an int indicating the conductivity, with
                    0 being nonconductive and 1 being conductive

                detection (int): an int indicating the sampled detection, the
                    int-detection mapping is available in utils.py

                weight (double): a double indicating the weight of the trash

            Returns:
                posteriors (list): a list of posterior probabilities of each trash category
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        category_prior = gtsam.DiscreteDistribution(self.P.Category, "200/300/250/200/50")
        likelihood_given_conductivity = [v for (k, v) in (self.P.get_pCT().likelihood(conductivity)).enumerate()]

        conductivity_factor = gtsam.DecisionTreeFactor(self.P.Category, likelihood_given_conductivity)
        detector_factor = gtsam.DecisionTreeFactor(self.P.Category, self.likelihood_given_detection(detection))
        weight_factor = gtsam.DecisionTreeFactor(self.P.Category, self.likelihood_given_weight(weight))

        likelihood = conductivity_factor * detector_factor * weight_factor

        posteriors = gtsam.DiscreteDistribution(likelihood * category_prior).pmf()
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return posteriors

    # TODO 16:
    ### DECISION ###
    def make_decision(self, posteriors):
        '''
        Returns the decision made by the robot given the likelihoods/posteriors you calculated

            Parameters:
                posteriors (list): a list of posteriors of each trash category

            Returns:
                action (int): an int indicating the action taken by the robot, the
                    int-action mapping is available in utils.py
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        ###############################################################################
        expected_cost = COST_TABLE @ posteriors
        action = np.argmin(expected_cost)
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return action


def main():
    rclpy.init()
    # Select the perception option
    robot = Robot(PerceptionOption.LIKELIHOOD_NO_SENSORS)
    rclpy.spin(robot)
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()