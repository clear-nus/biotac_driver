#!/usr/bin/env python

'''
Works only for TWO Biotacs.
Annoyingly written for myself such that it is hard to make it generalize to n biotacs.
First calibrate the biotac.
$rostopic pub -1 /biotac_sub std_msgs/String "calibrate"
Echo the following topic
$rostopic echo /biotac_pub_centered/bt_data
Calibration-centered numbers.
$rostopic echo /touched
Now touch the biotac.
Published array contains 8 numbers.
1st number is BioTac index, starting from 1.
2nd number is time.time().
Next 3 numbers in the array is for contact position.
Last 3 numbers in the array is for contact surface normal.

'''

import roslib; roslib.load_manifest('biotac_logger')
import rospy
import os, sys
from rosjson_time import rosjson_time
from types import *
import time
import numpy as np

from std_msgs.msg import String, Float64MultiArray
from biotac_sensors.msg import BioTacHand, SignedBioTacHand, SignedBioTacData
#from playsound import playsound
import matplotlib.pyplot as plt



class BIOTAC():
  def __init__(self, ID):
    self.ID = ID
    self.bt_serial=""
    self.bt_position=""
    self.calibrated = False #Detect touch only when calibrated once
    self.calibration_data_count = 0.0
    self.touch_counter = 0
    self.touch_threshold = 5 # Need N consecutive "abnormal" force readings to detect touch

    self.init_variables()
    self.confirmed_pdc_mu = -1.0 # Calibrated mean
    self.confirmed_pdc_var = -1.0
    self.confirmed_electrode_mu = np.zeros(19)
    self.confirmed_tac_mu = -1.0
    self.confirmed_tdc_mu = -1.0
    self.confirmed_pac_mu = np.zeros(22)
    self.electrode_normal = np.array([[0.196, -0.956, -0.220],
                           [0.0, -0.692, -0.722],
                           [0.0, -0.692, -0.722],
                           [0.0, -0.976, -0.220],
                           [0.0, -0.692, -0.722],
                           [0.0, -0.976, -0.220],
                           [0.5, 0.0, -0.866],
                           [0.5, 0.0, -0.866],
                           [0.5, 0.0, -0.866],
                           [0.5, 0.0, -0.866],
                           [0.196, 0.956, -0.220],
                           [0.0, 0.692, -0.722],
                           [0.0, 0.692, -0.722],
                           [0.0, 0.976, -0.220],
                           [0.0, 0.692, -0.722],
                           [0.0, 0.976, -0.220],
                           [0.0, 0.0, -1.000],
                           [0.0, 0.0, -1.000],
                           [0.0, 0.0, -1.000]])
    self.electrode_position = 0.001 * np.array([[0.993, -4.855, -1.116], #0.001 because vectors are in mm unit.
                             [-2.700, -3.513, -3.670],
                             [-6.200, -3.513, -3.670],
                             [-8.000, -4.956, -1.116],
                             [-10.500, -3.513, -3.670],
                             [-13.400, -4.956, -1.116],
                             [4.763, -0.000, -2.330],
                             [3.031, -1.950, -3.330],
                             [3.031, 1.950, -3.330],
                             [1.299, 0.000, -4.330],
                             [0.993, 4.855, -1.116],
                             [-2.700, 3.513, -3.670],
                             [-6.200, 3.513, -3.670],
                             [-8.000, 4.956, -1.116],
                             [-10.500, 3.513, -3.670],
                             [-13.400, 4.956, -1.116],
                             [-2.800, 0.000, -5.080],
                             [-9.800, 0.000, -5.080],
                             [-13.600, 0.000, -5.080]])
  def init_variables(self):
    self.pdc_mu_t = -1.0 #Online mean/var updates
    self.pdc_mu_t_1 = -1.0
    self.pdc_var_t = -1.0
    self.pdc_var_t_1 = -1.0
    self.electrode_mu_t = np.zeros(19)
    self.electrode_mu_t_1 = np.zeros(19)
    self.tac_mu_t = -1.0
    self.tac_mu_t_1 = -1.0
    self.tdc_mu_t = -1.0
    self.tdc_mu_t_1 = -1.0
    self.pac_mu_t = np.zeros(22)
    self.pac_mu_t_1 = np.zeros(22)


  def get_centered_data(self, bt_data):
    centered_data = SignedBioTacData()
    centered_data.bt_position = bt_data.bt_position
    centered_data.bt_serial = bt_data.bt_serial
    centered_data.tdc_data = bt_data.tdc_data - self.confirmed_tdc_mu
    centered_data.tac_data = bt_data.tac_data - self.confirmed_tac_mu
    centered_data.pdc_data  = bt_data.pdc_data - self.confirmed_pdc_mu
    centered_data.pac_data = np.array(bt_data.pac_data) - self.confirmed_pac_mu
    centered_data.electrode_data = np.array(bt_data.electrode_data) - self.confirmed_electrode_mu
    return centered_data

  def calibration_finished(self):
    self.confirmed_pdc_mu = self.pdc_mu_t
    self.confirmed_pdc_var = self.pdc_var_t
    self.confirmed_electrode_mu = self.electrode_mu_t
    self.confirmed_tac_mu = self.tac_mu_t
    self.confirmed_tdc_mu = self.tdc_mu_t
    self.confirmed_pac_mu = self.pac_mu_t
    self.calibration_data_count = 0
    self.init_variables()

  def calibrate(self, bt_data):
    self.calibration_data_count += 1
    if self.calibration_data_count == 1:
      self.pdc_mu_t_1 = bt_data.pdc_data
      self.pdc_var_t_1 = 0
      self.electrode_mu_t_1 = np.array(bt_data.electrode_data)
      self.tac_mu_t_1 = np.array(bt_data.tac_data)
      self.tdc_mu_t_1 = np.array(bt_data.tdc_data)
      self.pac_mu_t_1 = np.array(bt_data.pac_data)
    else: #online mean and var update formula
      self.pdc_mu_t = (self.calibration_data_count - 1.0)/self.calibration_data_count * float(self.pdc_mu_t_1) + 1.0 / self.calibration_data_count * float(bt_data.pdc_data)
      self.pdc_var_t = (self.pdc_var_t_1*(self.calibration_data_count - 2.0) + (bt_data.pdc_data - self.pdc_mu_t_1)*(bt_data.pdc_data - self.pdc_mu_t))/(self.calibration_data_count - 1.0)
      self.pdc_mu_t_1 = self.pdc_mu_t
      self.pdc_var_t_1 = self.pdc_var_t
      self.electrode_mu_t = (self.calibration_data_count - 1.0)/self.calibration_data_count * self.electrode_mu_t_1 + 1.0 / self.calibration_data_count * np.array(bt_data.electrode_data)
      self.electrode_mu_t_1 = self.electrode_mu_t
      self.tac_mu_t = (self.calibration_data_count - 1.0)/self.calibration_data_count * self.tac_mu_t_1 + 1.0 / self.calibration_data_count * np.array(bt_data.tac_data)
      self.tac_mu_t_1 = self.tac_mu_t
      self.tdc_mu_t = (self.calibration_data_count - 1.0)/self.calibration_data_count * self.tdc_mu_t_1 + 1.0 / self.calibration_data_count * np.array(bt_data.tdc_data)
      self.tdc_mu_t_1 = self.tdc_mu_t
      self.pac_mu_t = (self.calibration_data_count - 1.0)/self.calibration_data_count * self.pac_mu_t_1 + 1.0 / self.calibration_data_count * np.array(bt_data.pac_data)
      self.pac_mu_t_1 = self.pac_mu_t
      print("Calibrating (ID: " + str(self.ID) + "), sample variance: "+str(self.pdc_var_t_1))

  def check_against_calibration(self, bt_data):
    if bt_data.pdc_data > (self.confirmed_pdc_mu + 50*self.confirmed_pdc_var**0.5):
      self.touch_counter += 1
      if self.touch_counter >= self.touch_threshold:
        weights = np.expand_dims(self.compute_weights(bt_data), axis = 0)
        signed_weights = np.expand_dims(self.compute_positive_electrode_weights(bt_data), axis = 0)
        # Unsigned compression + depression is from the paper, but is NOT WORKING WELL.
        # print("Unsigned/Compression+Depression")
        # print(-np.sort(-weights))
        # print(np.argsort(-weights)+1)
        #print("Negative-only/Compression")
        #print(-np.sort(-signed_weights))
        #print(np.argsort(-signed_weights)+1)
        # touch_position = self.compute_touch_position(weights)
        # touch_normal = self.compute_touch_normal(weights)
        # touch_data = np.concatenate((touch_position,touch_normal),axis=None)
        signed_touch_position = self.compute_touch_position(signed_weights)
        signed_touch_normal = self.compute_touch_normal(signed_weights)
        signed_touch_data = np.concatenate((self.ID+1, time.time(), signed_touch_position, signed_touch_normal),axis=None)
        # everything_data = np.concatenate((touch_data,signed_touch_data),axis=None)
        return Float64MultiArray(data=signed_touch_data)
    else:
      self.touch_counter = 0
      return -1

  def compute_positive_electrode_weights(self,bt_data):
    weight_power = 2
    return(np.power(np.min((np.zeros(19),np.array(bt_data.electrode_data)-self.confirmed_electrode_mu),axis=0),weight_power)/np.sum(np.power(np.min((np.zeros(19),np.array(bt_data.electrode_data)-self.confirmed_electrode_mu),axis=0),weight_power)))

  def compute_weights(self,bt_data):
    weight_power = 2
    return(np.power(np.array(bt_data.electrode_data)-self.confirmed_electrode_mu,weight_power)/np.sum(np.power(np.array(bt_data.electrode_data)-self.confirmed_electrode_mu,weight_power)))

  def compute_touch_position(self,weights):
    return np.matmul(weights,self.electrode_position)

  def compute_touch_normal(self,weights):
    return np.matmul(weights,self.electrode_normal)


class BioTacListener:

  def __init__(self):
    self.biotacs = {}
    self.calibrating = False
    self.setup_done = False
    self.n_biotacs = -1
    rospy.init_node('biotac_json_logger')
    self.end_time = -1 # calibration time set by command
    self.last_calibrated = -1 
    self.calibrated = False
    self.calibration_time = 3

  # Called each time there is a new message
  def biotacCallback(self,data):

    # setup biotacs
    if self.setup_done == False:
      for i in range(len(data.bt_data)):
        self.biotacs[i] = BIOTAC(i)
      self.setup_done=True
      self.n_biotacs = i+1

    data.header.frame_id = ''

    if self.end_time != -1 and self.calibrating:
      if time.time() >= self.end_time:
        finish_msg = '%f seconds calibration complete!' % self.calibration_time
        for i in range(self.n_biotacs):
          self.biotacs[i].calibration_finished()
        rospy.loginfo(finish_msg)
        self.last_calibrated = time.time()
        self.calibrated = True
        self.end_time = -1
        self.calibrating = False
      else:
        for i in range(self.n_biotacs):
          self.biotacs[i].calibrate(data.bt_data[i])


    elif self.calibrated:
      #self.check_against_calibration(data)
      for i in range(self.n_biotacs):
        #self.biotacs[i].check_against_calibration(data.bt_data[i])
        sensor_info = self.biotacs[i].check_against_calibration(data.bt_data[i])
        if (not (sensor_info == -1)) and (not (sensor_info == None)):
          self.pub.publish(sensor_info)
      self.publish_centered_data(data)

  def publish_centered_data(self, data):
    # publish calibrated data
    centered_data = SignedBioTacHand()
    centered_data.header.stamp.secs = data.header.stamp.secs
    centered_data.header.stamp.nsecs = data.header.stamp.nsecs
    for i in range(self.n_biotacs):
      centered_data.bt_data.append(self.biotacs[i].get_centered_data(data.bt_data[i]))
    self.pub_centered.publish(centered_data)

  # listen to commands
  def commandCallback(self, data):
    if data.data == 'calibrate':
      if self.calibrating:
        print("Already calibrating")
      else:
        print("Calibrating for next "+str(self.calibration_time)+" seconds!")
        self.end_time = time.time() + self.calibration_time
        self.calibrating = True
    elif data.data == 'last_calibrated':
      if self.last_calibrated == -1:
        print("Never calibrated!")
      else:
        print("Calibrated "+str(time.time()-self.last_calibrated)+" seconds ago!")
    else:
      print("Bad command!")

  # Setup the subscriber Node
  def listener(self):
    self.pub = rospy.Publisher('touched', Float64MultiArray, queue_size=10)
    self.pub_centered = rospy.Publisher('biotac_pub_centered', SignedBioTacHand, queue_size=10)
    #self.pub_centered_1 = rospy.Publisher('biotac_pub_centered1', SignedBioTacData, queue_size=10)
    #self.pub_centered_2 = rospy.Publisher('biotac_pub_centered2', SignedBioTacData, queue_size=10)
    rospy.Subscriber('biotac_pub', BioTacHand, self.biotacCallback,queue_size=1000)
    rospy.Subscriber('biotac_sub', String, self.commandCallback,queue_size=1000)
    rospy.spin()

if __name__ == '__main__':
  bt_listener = BioTacListener()
  bt_listener.listener()
