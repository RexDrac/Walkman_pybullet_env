import numpy as np
import math
from sigmaban_gym_env.Angle import *

class JointModel:
    def __init__(self, name):
        self._name = name
        self._featureBacklash = False
        self._featureFrictionStribeck = True
        self._featureReadDiscretization = False
        self._featureOptimizationVoltage = False
        self._featureOptimizationResistance = True
        self._featureOptimizationRegularization = False
        self._featureOptimizationControlGain = False
        
        self._goalTime = 0.0
        self._goalHistory = []
        self._isInitialized = False
        #Backlash initial state
        self._stateBacklashIsEnabled = False
        self._stateBacklashPosition = 0.0
        self._stateBacklashVelocity = 0.0

        #Firmware coefficients
        self._coefAnglePosToPWM = (4096.0/(2.0*math.pi))/3000.0
        self._coefPWMBound = 0.983

        #Default parameters value
        #Friction static regularization coefficient
        self._paramFrictionRegularization = 20.0
        #Friction velocity limit
        self._paramFrictionVelLimit = 0.2
        #Joint internal gearbox inertia
        self._paramInertiaIn = 0.003
        #Friction internal gearbox parameters
        #BreakIn is a positive offset above CoulombIn
        self._paramFrictionViscousIn = 0.07
        self._paramFrictionBreakIn = 0.03
        self._paramFrictionCoulombIn = 0.05
        #Joint external gearbox inertia
        self._paramInertiaOut = 0.001
        #Friction external gearbox parameters
        #BreakOut is a positive offset above CoulombOut
        self._paramFrictionViscousOut = 0.01
        self._paramFrictionBreakOut = 0.03
        self._paramFrictionCoulombOut = 0.02
        #Electric motor voltage
        self._paramElectricVoltage = 12.5
        #Electric motor ke
        self._paramElectricKe = 1.4
        #Electric motor resistance
        self._paramElectricResistance = 4.07
        #Motor proportional control
        self._paramControlGainP = 30.0
        #Motor position discretization coefficient
        self._paramControlDiscretization = 2048.0
        #Control lag in seconds
        self._paramControlLag = 0.030
        #Backlash enable to disable position threshold
        #Positive offset above activation
        self._paramBacklashThresholdDeactivation = 0.01
        #Backlash disable to enable position threshold
        self._paramBacklashThresholdActivation = 0.02
        #Backlash maximum angular distance
        self._paramBacklashRangeMax = 0.05

        #Initialize the parameters according to selected feature model
        tmpParams = self.getParameters()
        self.setParameters(tmpParams)

    def getName(self):
        return self._name

    def getParameters(self):
        index = 0
        #Build parameters vector
        parameters = list()#create list for parameters

        #External friction parameters
        parameters.append(self._paramFrictionViscousOut)
        parameters.append(self._paramFrictionCoulombOut)
        parameters.append(self._paramInertiaOut)
        index+=3
        if self._featureFrictionStribeck:
            parameters.append(self._paramFrictionVelLimit)
            parameters.append(self._paramFrictionBreakOut)
            index+=2

        #Internal friction parameters
        if self._featureBacklash:
            parameters.append(self._paramFrictionViscousIn)
            parameters.append(self._paramFrictionCoulombIn)
            parameters.append(self._paramInertiaIn)
            index+=3
            if self._featureFrictionStribeck:
                parameters.append(self._paramFrictionBreakIn)
                index+=1

        #Backlash internal
        if self._featureBacklash:
            parameters.append(self._paramBacklashRangeMax)
            parameters.append(self._paramBacklashThresholdDeactivation)
            parameters.append(self._paramBacklashThresholdActivation)
            index+=3

        #Other control and electric parameters
        parameters.append(self._paramControlLag)
        parameters.append(self._paramElectricKe)
        index+=2

        #Read Position encoder discretization
        if self._featureReadDiscretization:
            parameters.append(self._paramControlDiscretization)
            index+=1

        #Electric power voltage
        if self._featureOptimizationVoltage:
            parameters.append(self._paramElectricVoltage)
            index+=1
        
        #Electric resistance
        if self._featureOptimizationResistance:
            parameters.append(self._paramElectricResistance)
            index+=1
        
        #Friction force refularization coefficient
        if self._featureOptimizationRegularization:
            parameters.append(self._paramFrictionRegularization)
            index+=1

        # Proportional control gain
        if self._featureOptimizationControlGain:
            parameters.append(self._paramControlGainP)
            index+=1

        return np.array(parameters) #convert to array

    def setParameters(self, parameters):
        tmpParams = np.array(parameters)
        for i in range(len(tmpParams)):
            if tmpParams[i] < 0.0:
                tmpParams[i] = 0.0

        #Read parameters vector
        #External friction parameters
        index = 0
        if len(tmpParams) < index + 3:
            print('JointModel invalid parameters size: ' + str(len(tmpParams)))
            print('JointModel desired parameters size: ' + str(index + 3))
            return
        
        self._paramFrictionViscousOut = tmpParams[index]
        self._paramFrictionCoulombOut = tmpParams[index+1]
        self._paramInertiaOut = tmpParams[index+2]
        index+=3

        if self._featureFrictionStribeck:
            if len(tmpParams) < index + 2:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 2))
                return
            self._paramFrictionVelLimit = tmpParams[index]
            self._paramFrictionBreakOut = tmpParams[index+1]
            index+=2

        #Internal friction parameters
        if self._featureBacklash:
            if len(tmpParams) < index + 3:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 3))
                return
            self._paramFrictionViscousIn = tmpParams[index]
            self._paramFrictionCoulombIn = tmpParams[index+1]
            self._paramInertiaIn = tmpParams[index+2]
            index+=3
            if self._featureFrictionStribeck:
                if len(tmpParams) < index + 1:
                    print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                    print('JointModel desired parameters size: ' + str(index + 1))
                    return
                self._paramFrictionBreakIn = tmpParams[index]
                index+=1
        
        #Backlash internal
        if self._featureBacklash:
            if len(tmpParams) < index + 3:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 3))
                return
            self._paramBacklashRangeMax = tmpParams[index]
            self._paramBacklashThresholdDeactivation = tmpParams[index+1]
            self._paramBacklashThresholdActivation = tmpParams[index+2]
            index+=3

        #Other control and electric parameters
        if len(tmpParams) < index + 2:
            print('JointModel invalid parameters size: ' + str(len(tmpParams)))
            print('JointModel desired parameters size: ' + str(index + 2))
            return        
        self._paramControlLag = tmpParams[index]
        self._paramElectricKe = tmpParams[index+1]
        index+=2

        #read position encoder discretization
        if self._featureReadDiscretization:
            if len(tmpParams) < index + 1:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 1))
                return
            self._paramControlDiscretization = tmpParams[index]
            index+=1
        
        #Electric power voltage
        if self._featureOptimizationVoltage:
            if len(tmpParams) < index + 1:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 1))
                return
            self._paramElectricVoltage = tmpParams[index]
            index+=1
        
        #Electric resistance
        if self._featureOptimizationResistance:
            if len(tmpParams) < index + 1:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 1))
                return
            self._paramElectricResistance = tmpParams[index]
            index+=1

        #Friction force regularization coefficient
        if self._featureOptimizationRegularization:
            if len(tmpParams) < index + 1:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 1))
                return
            self._paramFrictionRegularization = tmpParams[index]
            index+=1

        #Propotional control gain
        if self._featureOptimizationControlGain:
            if len(tmpParams) < index + 1:
                print('JointModel invalid parameters size: ' + str(len(tmpParams)))
                print('JointModel desired parameters size: ' + str(index + 1))
                return
            self._paramControlGainP = tmpParams[index]
            index+=1

        return

    def getInertia(self):
        return
    
    def setMaxVoltage(self, voltage):
        return

    def getMaxVoltage(self):
        return

    def frictionTorque(self, vel):
        if self._stateBacklashIsEnabled:
            return self.computeFrictionTorque(vel, 0, False, True)
        else:
            return self.computeFrictionTorque(vel, 0, True, True)
        return

    def controlTorque(self, pos, vel):
        if self._stateBacklashIsEnabled:
            return 0.0
        else:
            return self.computeControlTorque(pos, vel)
        return
    
    def updateState(self, dt, goal, pos, vel):
        #Hidden state intialization
        if not self._isInitialized:
            self._goalTime = 0.0
            # if not self._goalHistory: #if not empty
            #     self._goalHistory = []
            self._goalHistory = []
            self._stateBacklashIsEnabled = False
            self._stateBacklashPosition = pos
            self._stateBacklashVelocity = vel
            self._isInitialized = True

        #Append given goal
        self._goalHistory.append([self._goalTime, goal])
        self._goalTime += dt 

        #Pop history to get current goal lag
        while ((len(self._goalHistory) >=2) and (self._goalHistory[0][0]<self._goalTime - self._paramControlLag)):
            self._goalHistory.pop(0)#pop first element, oldest element

        #Update backlash model
        if self._featureBacklash:
            #Compute backlash acceleration
            backlashControlTorque = self.computeControlTorque(pos, vel)
            backlashFrictionTorque = self.computeFrictionTorque(vel, backlashControlTorque, True, False)
            backlashAcc = (backlashControlTorque-backlashFrictionTorque)/self._paramInertiaIn
            #Update backlash velocity and position
            backlashNextVel = self._stateBacklashVelocity + dt*backlashAcc
            self._stateBacklashVelocity = 0.5*self._stateBacklashVelocity + 0.5*backlashNextVel #running average filter
            self._stateBacklashPosition = self._stateBacklashPosition + dt*self._stateBacklashVelocity
            #Update backlash state
            relativePos = abs(AngleDistance(pos, self._stateBacklashPosition))
            #Used state transition thresholds
            usedThresholdDeactivation = self._paramBacklashThresholdDeactivation + self._paramBacklashThresholdActivation
            usedThresholdActivation = self._paramBacklashThresholdActivation
            if self._stateBacklashIsEnabled and relativePos > usedThresholdDeactivation:
                self._stateBacklashIsEnabled = False
            elif (not self._stateBacklashIsEnabled) and relativePos < usedThresholdActivation:
                self._stateBacklashIsEnabled = True
            #Bound backlash position
            if AngleDistance(pos, self._stateBacklashPosition) >= self._paramBacklashRangeMax:
                self._stateBacklashPosition = pos + self._paramBacklashRangeMax
                self._stateBacklashVelocity = 0.0
            if AngleDistance(pos, self._stateBacklashPosition) <= self._paramBacklashRangeMax:
                self._stateBacklashPosition = pos - self._paramBacklashRangeMax
                self._stateBacklashVelocity = 0.0

            self._stateBacklashPosition = AngleBound(self._stateBacklashPosition)

    def getDelayedGoal(self):
        if len(self._goalHistory) == 0:
            return
        else:
            return self._goalHistory[0][1]

    def getBacklashStateEnabled(self):
        return

    def getBacklashStatePos(self):
        return

    def getBacklashStateVel(self):
        return

    def resetHiddenState(self):
        self._isInitialized = False
        self._goalTime = 0.0
        # while(not self._goalHistory):
        #     self._goalHistory.pop(0)
        self._goalHistory = []#clear history
        self._stateBacklashIsEnabled = True
        self._stateBacklashPosition = 0.0
        self._stateBacklashVelocity = 0.0

    def boundState(self, pos, vel):
        #Check numerical instability
        if abs(pos>1e10 or vel>1e10):
            print('JointModel numerical instability'
             + ' name=' + str(self._name)
             + ' pos=' + str(pos)
             + ' vel=' + str(vel))

        #Bound position angle inside [-pi, pi]
        pos = AngleBound(pos)
        return pos
    
    def computeElectricTension(self, velGoal, accGoal, torqueGoal):
        #Compute friction torque
        #Backlash is not considered
        frictionTorque = self.computeFrictionTorque(velGoal, torqueGoal, True, True)
        #Compute torque from gears inertia
        #backlash is not considered
        usedInertia = 0.0
        if not self._featureBacklash:
            usedInertia = self._paramInertiaOut
        else:
            usedInertia = self._paramInertiaIn + self._paramInertiaOut
        
        inertiaTorque = accGoal*usedInertia

        #Compute internal torque seen by the motor
        #to produce expected motion
        torqueInternalGoal = torqueGoal - frictionTorque + inertiaTorque

        #Compute expected electric tension to produce needed torque at motor output
        tensionGoal = torqueInternalGoal*self._paramElectricResistance/self._paramElectricKe + velGoal*self._paramElectricResistance

        return tensionGoal

    def computeFeedForward(self, velGoal, accGoal, torqueGoal):
        #Compute expected motor tension
        tensionGoal = self.computeElectricTension(velGoal, accGoal, torqueGoal)

        #Compute expected control PWN ratio
        controlRatioGoal = tensionGoal/self._paramElectricVoltage

        #Bound control ratio to controller capacity
        if controlRatioGoal > self._coefPWMBound:
            tensionGoal = self._coefPWMBound
        if tensionGoal < -self._coefPWMBound:
            tensionGoal = -self._coefPWMBound
        
        #Compute angular offset from tension
        angularOffset = controlRatioGoal/(self._paramControlGainP*self._coefAnglePosToPWM)

        return angularOffset

    def printParameters(self):
        return
    
    def computeFrictionTorque(self, vel, torque, isInFriction, isOutFriction):
        #Apply internal and external friction model
        usedFrictionVelLimit = self._paramFrictionVelLimit
        usedFrictionViscous = 0.0
        usedFrictionBreak = 0.0
        usedFrictionCoulomb = 0.0
        if isInFriction:
        # if isInFriction and self._featureBacklash:
            usedFrictionViscous += self._paramFrictionViscousIn
            usedFrictionBreak += self._paramFrictionBreakIn + self._paramFrictionCoulombIn
            usedFrictionCoulomb += self._paramFrictionCoulombIn
        if isOutFriction:
            usedFrictionViscous += self._paramFrictionViscousOut
            usedFrictionBreak += self._paramFrictionBreakOut + self._paramFrictionCoulombOut
            usedFrictionCoulomb += self._paramFrictionCoulombOut
        if not self._featureFrictionStribeck:
            usedFrictionBreak = usedFrictionCoulomb

        #Compute friction
        beta = math.exp(-abs(vel/usedFrictionVelLimit))
        forceViscous = -usedFrictionViscous*vel
        forceStatic1 = -beta*usedFrictionBreak
        forceStatic2 = -(1.0-beta)*usedFrictionCoulomb
        #Static friction regularization passing by zero to prevent too stiff dynamics and allows for nice continuous forces and accelerations
        forceStaticRegularized = (forceStatic1 + forceStatic2)*math.tanh(self._paramFrictionRegularization*vel)
        return forceViscous + forceStaticRegularized

    def computeControlTorque(self, pos, vel):
        #Apply position discretization
        discretizedPos = pos
        if self._featureReadDiscretization:
            motorStepCoef = math.pi / self._paramControlDiscretization
            discretizedPos = math.floor(pos/motorStepCoef)*motorStepCoef

        #Retrieve delayed goal
        delayedGoal = self.getDelayedGoal()

        #Angular distance in radian
        error = AngleDistance(delayedGoal, discretizedPos)
        # print('bounded angle distance', AngleDistance(delayedGoal, discretizedPos))
        # print('unbounded angle distance', delayedGoal - discretizedPos)
        #Compute Motor control PWM ratio
        controlRatio = -self._paramControlGainP*error*self._coefAnglePosToPWM
        # controlRatio = self._paramControlGainP*(delayedGoal-discretizedPos)*self._coefAnglePosToPWM
        #Bound the PWM control ratio between -1.0*_coefPWMBound and 1.0*_coefPWMBound
        # controlRatio = np.clip(controlRatio, -self._coefPWMBound, self._coefPWMBound)
        if controlRatio > self._coefPWMBound:
            controlRatio = self._coefPWMBound
        elif controlRatio < -self._coefPWMBound:
            controlRatio = -self._coefPWMBound
        # print('controlRatio',controlRatio)
        #Compute the applied tension on the electric motor by H-bridge
        tension = controlRatio*self._paramElectricVoltage

        #Compute applied electric torque
        # torque = tension*self._paramElectricKe/self._paramElectricResistance - vel*math.pow(self._paramElectricKe, 2)/self._paramElectricResistance
        #'kp = 23.54', self._paramControlGainP*self._coefAnglePosToPWM*self._paramElectricVoltage*self._paramElectricKe/self._paramElectricResistance
        #'kd = 0.48', math.pow(self._paramElectricKe, 2)/self._paramElectricResistance
        motor_torque = tension*self._paramElectricKe/self._paramElectricResistance
        v_torque = - vel*math.pow(self._paramElectricKe, 2)/self._paramElectricResistance
        torque = motor_torque + v_torque
        if motor_torque > 0.0:
            torque = np.clip(torque,0.0,motor_torque)
        elif motor_torque < 0.0:
            torque = np.clip(torque,motor_torque,0.0)

        return torque
