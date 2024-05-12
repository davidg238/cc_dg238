class BaseController:
  def update(self, target_lataccel, current_lataccel, state):
    raise NotImplementedError


class OpenController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return target_lataccel


class SimpleController(BaseController):
  def update(self, target_lataccel, current_lataccel, state):
    return (target_lataccel - current_lataccel) * 0.3

class PidController(BaseController):
  """ This class implements a PID controller.
  """

  def __init__(self, Kp=2, Ki=.02, Kd=0.1, Kaw=0.025, T_C=20, max=2, min=-2, max_rate=1.0):
      self.Kp = Kp  # Proportional gain
      self.Ki = Ki  # Integral gain
      self.Kd = Kd  # Derivative gain
      self.Kaw = Kaw  # Anti-windup gain
      self.T_C = T_C  # Time constant for derivative filtering
      # self.T = T  # Time step
      self.max = max # Maximum command
      self.min = min  # Minimum command
      self.max_rate = max_rate  # Maximum rate of change of the command
      self.integral = 0  # Integral term
      self.err_prev = 0  # Previous error
      self.deriv_prev = 0  # Previous derivative
      self.command_sat_prev = 0  # Previous saturated command
      self.command_prev = 0  # Previous command
      self.command_sat = 0  # Current saturated command

      self.time_last = time.time()

  def update(self, setpoint, measurement, state):
      """ Execute a step of the PID controller.

      Inputs:
          measurement: current measurement of the process variable
          setpoint: desired value of the process variable
      """
      time_now = time.time()
      T = time_now - self.time_last
      # Calculate error
      err = setpoint - measurement

      # Update integral term with anti-windup
      self.integral += self.Ki * err * T + self.Kaw * (self.command_sat_prev - self.command_prev) * T
      # self.integral += self.Ki * err * T # + self.Kaw * (self.command_sat_prev - self.command_prev) * T

      # Calculate filtered derivative
      deriv_filt = (err - self.err_prev + self.T_C * self.deriv_prev) / (T + self.T_C)
      self.err_prev = err
      self.deriv_prev = deriv_filt

      # Calculate command using PID equation
      command = self.Kp * err + self.integral + self.Kd * deriv_filt

      # Store previous command
      self.command_prev = command

      # Saturate command
      if command > self.max:
          self.command_sat = self.max
      elif command < self.min:
          self.command_sat = self.min
      else:
          self.command_sat = command

      # Apply rate limiter
      if self.command_sat > self.command_sat_prev + self.max_rate * T:
          self.command_sat = self.command_sat_prev + self.max_rate * T
      elif self.command_sat < self.command_sat_prev - self.max_rate * T:
          self.command_sat = self.command_sat_prev - self.max_rate * T

      # Store previous saturated command
      self.command_sat_prev = self.command_sat
      self.time_last = time_now

      return self.command_sat


import time
class Pid2Controller(BaseController):
  # controlling the lateral acceleration
  def __init__(self, kp=0.2, ki=0.01, kd=0.05, t_c=0.01):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.T_C = t_c  # Time constant for derivative filtering
    self.ks = -1 # reverse acting controller
    self.spio =  False # integral action only on SP change

    self.out_max = 2
    self.out_min = -2

    self.prev_err = 0
    self.integral = 0
    self.dev_last = 0
    self.pv_last = 0
    self.out_last = 0
    self.deriv_prev = 0
    self.deriv_adj_prev = 0
    self.pv_deriv_last = 0

    self.time_last = time.time()

  def tune(self, kp, ki, kd, t_c):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.T_C = t_c

  def update(self, sp, pv, state):
    time_now = time.time()
    dT = time_now - self.time_last
    dev = pv - sp
    p1 = dev - self.dev_last
    p2 = pv - self.pv_last
    if self.spio:
      p3 = p2
    else:
      p3 = p1
    proportional = p3 * self.ks * self.kp
    if self.ki != 0:
      integral  = dev * self.ks * self.kp * (dT / self.ki)
    else:
      integral = 0
    pv_deriv = p2 * self.kd  + self.pv_deriv_last * (self.T_C / (dT + self.T_C))
    pv_deriv_adj = pv_deriv * self.ks * self.kp
    deriv = pv_deriv_adj - self.deriv_adj_prev
    out_pid = proportional + integral + deriv + self.out_last
    out = min( max(self.out_min, out_pid), self.out_max)
    self.pv_last = pv
    self.dev_last = dev
    self.out_last = out
    self.time_last = time_now
    self.deriv_prev = deriv
    self.deriv_adj_prev = pv_deriv_adj
    self.pv_deriv_last = pv_deriv

    return out

# python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple

# python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller pid --baseline_controller simple


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyController(BaseController):

  def __init__(self):
    self.eo = 0
# New Antecedent/Consequent objects hold universe variables and membership functions
    self.error = ctrl.Antecedent(np.arange(0, 11, 1), 'error')
    self.error_change = ctrl.Antecedent(np.arange(0, 11, 1), 'error_change')
    self.steer = ctrl.Consequent(np.arange(0, 26, 1), 'steer')
# Auto-membership function population is possible with .automf(3, 5, or 7)
    self.error.automf(3)
    self.error_change.automf(3)
# Custom membership functions
    self.steer['low'] = fuzz.trimf(self.steer.universe, [0, 0, 13])
    self.steer['medium'] = fuzz.trimf(self.steer.universe, [0, 13, 25])
    self.steer['high'] = fuzz.trimf(self.steer.universe, [13, 25, 25])

    self.rule1 = ctrl.Rule(self.target['poor'] | self.current['poor'], self.steer['low'])
    self.rule2 = ctrl.Rule(self.current['average'], self.steer['medium'])
    self.rule3 = ctrl.Rule(self.current['good'] | self.target['good'], self.steer['high'])

    self.steer_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3])
    self.steering = ctrl.ControlSystemSimulation(self.steer_ctrl)

  def update(self, target_lataccel, current_lataccel, state):

    e = target_lataccel - current_lataccel
    ec = self.eo - e
    self.steering.input['error'] = e
    self.steering.input['error_change'] = ec
    self.steering.compute()
    self.eo = e
    return self.steering.output['steer']



CONTROLLERS = {
  'open': OpenController,
  'simple': SimpleController,
  'pid': PidController,
  'pid2': Pid2Controller,
  'fuzzy': FuzzyController,
}