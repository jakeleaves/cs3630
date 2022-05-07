import gtsam
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int16
from utils import Bin, Detection, Probability, Trash, COST_TABLE

np.random.seed(3630)


class World(Node):
    '''
    This class simulates the world, providing sensor readings and metadata

        PARAMETERS
            future (rclpy.task.Future): Represents the outcome of task and in our case is used to know when the
                simulation should complete
            num_iterations (int): The number of pieces of trash the world should simulate
    '''

    def __init__(self, future, num_iterations=20):
        if future is not None:
            super().__init__('world_publisher')

        self.P = Probability()

        self.last_category = None
        self.waiting_for_response = False
        self.num_iterations = num_iterations
        self.cost = 0

        if future is not None:
            self.ros_init(future)

    def ros_init(self, future):
        self.future = future

        # Create publishers
        self.iteration_publisher = self.create_publisher(Int16, 'iteration', 1)
        self.get_logger().info('iteration_publisher has been started.')

        self.conductivity_publisher = self.create_publisher(Bool, 'conductivity', 1)
        self.get_logger().info('conductivity_publisher has been started.')

        self.detection_publisher = self.create_publisher(Int16, 'detection', 1)
        self.get_logger().info('detection_publisher has been started.')

        self.weight_publisher = self.create_publisher(Float32, 'weight', 1)
        self.get_logger().info('weight_publisher has been started.')

        # Create subscriber
        self.decision_subscriber = self.create_subscription(Int16, 'decision', self.subscriber_callback, 1)
        self.get_logger().info('decision_subscriber has been started.')

        self.iteration = 0
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publisher_callback)

    def sample_category(self):
        '''
        Returns a sample of trash category by sampling with the prior probabilities
        of the trash categories

            Parameters:
                None

            Returns:
                sample (int): an int indicating the sampled trash category, the
                    int-category mapping is available in utils.py
        '''
        sample = None
        ###############################################################################
        #                             START OF YOUR CODE                              #
        # TODO 3                                                                      #
        ###############################################################################
        cdf = np.cumsum(self.P.get_category_prior_pmf())
        u = np.random.rand()
        for sample in range(5):
            if u < float(cdf[sample]):
                return sample

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return sample

    def sample_conductivity(self, category=None):
        '''
        Returns a sample of conductivity using the conditional probability
        given the trash category.

            Parameters:
                category (int): an int indicating the trash category

            Returns:
                conductivity (int): an int indicating the conductivity, with
                    0 being nonconductive and 1 being conductive
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        # TODO 8                                                                      #
        ###############################################################################
        values = gtsam.DiscreteValues()
        values[self.P.Category[0]] = category
        conductivity = self.P.get_pCT().sample(values)

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return conductivity

    def sample_detection(self, category=None):
        '''
        Returns a sample of detection using the conditional probability given
        the trash category.

            Parameters:
                category (int): an int indicating the trash category

            Returns:
                detection (int): an int indicating the sampled detection, the
                    int-detection mapping is available in utils.py
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        # TODO 9                                                                      #
        ###############################################################################
        values = gtsam.DiscreteValues()
        values[self.P.Category[0]] = category
        detection = self.P.get_pDT().sample(values)
        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return detection

    def sample_weight(self, category=None):
        '''
        Returns a sample of weight using the conditional probability given
        the trash category.

            Parameters:
                category (int): an int indicating the trash category

            Returns:
                weight (double): a double indicating the sampled weight
        '''

        ###############################################################################
        #                             START OF YOUR CODE                              #
        # TODO 10                                                                     #
        ###############################################################################

        weight = np.random.normal(*(self.P.get_pWT()[category]))

        ###############################################################################
        #                              END OF YOUR CODE                               #
        ###############################################################################
        return weight

    def publisher_callback(self):
        if not self.waiting_for_response:
            if self.iteration >= self.num_iterations:
                self.future.set_result('finished')
                return
            # Sample sensors
            self.last_category = self.sample_category()
            conductivity_sample = self.sample_conductivity(self.last_category)
            detection_sample = self.sample_detection(self.last_category)
            weight_sample = self.sample_weight(self.last_category)

            # Create messages
            iteration_msg = Int16()
            iteration_msg.data = self.iteration
            conductivity_msg = Bool()
            conductivity_msg.data = conductivity_sample > 0
            detection_msg = Int16()
            detection_msg.data = detection_sample
            weight_msg = Float32()
            weight_msg.data = weight_sample

            # Publish samples
            self.iteration_publisher.publish(iteration_msg)
            self.conductivity_publisher.publish(conductivity_msg)
            self.detection_publisher.publish(detection_msg)
            self.weight_publisher.publish(weight_msg)
            self.get_logger().info(
                f'Sample {self.iteration}:' +
                f'\tActual: {Trash(self.last_category).name};'
                f'\tConductivity: {conductivity_sample > 0};' +
                f'\tDetection: {Detection(detection_sample).name};' +
                f'\tWeight: {weight_sample} grams.'
            )
            self.iteration += 1

            self.waiting_for_response = True

    def subscriber_callback(self, msg):
        cost = COST_TABLE[msg.data][self.last_category]
        self.get_logger().info(
            f'The robot decided to put the {Trash(self.last_category).name} in the {Bin(msg.data).name} bin resulting in a cost of {cost}.')
        self.cost += cost
        self.get_logger().info(f'Total cost: {self.cost}')

        self.waiting_for_response = False


def main():
    rclpy.init()
    future = rclpy.task.Future()
    world = World(future, num_iterations=20)
    rclpy.spin_until_future_complete(world, future)
    world.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
