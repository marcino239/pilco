import scipy.integrate as integrate
import numpy as np

from numpy import sin, cos, pi, zeros_like


## Define a few constants
G =  9.8      # acceleration due to gravity, in m/s^2
L1 = 1.0      # length of pendulum 1 in m
L2 = 1.0      # length of pendulum 2 in m
M1 = 1.0      # mass of pendulum 1 in kg
M2 = 1.0      # mass of pendulum 2 in kg
d1 = 0.5 # friction for joint 1 in rad / s
d2 = 0.2 # friction for joint 2 in rad / s

rad = pi/180  # radians in 1 deg

def derivs(state, t):
	'''
		state[0] - joint 1 angle
		state[1] - joint 1 speed
		state[2] - joint 2 angle
		state[3] - joint 2 speed
	'''

	dxdt = zeros_like(state)   # init. dervative array
	del_ = state[2] - state[0]    # difference of angle1 and angle2

	dxdt[0] = state[1]            # derv. of angle1

	den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)  # deno. of dxdt[2]
	dxdt[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
				+ M2*G*sin(state[2])*cos(del_) 
				+ M2*L2*state[3]*state[3]*sin(del_)
				- (M1+M2)*G*sin(state[0]))/den1 - d1 * state[1]

	dxdt[2] = state[3]  # derv. of angle2

	den2 = (L2/L1)*den1  # deno. of dxdt[3]
	dxdt[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)  
				+ (M1+M2)*G*sin(state[0])*cos(del_)
				- (M1+M2)*L1*state[1]*state[1]*sin(del_)
				- (M1+M2)*G*sin(state[2]))/den2 - d2 * state[3]

	return dxdt  # return time derv. array
    
    
class Pendubot:
	def __init__( self, state, ts = 0.001 ):
		'''
			ts - sampling time
		'''
		self.state = state
		self.ts = ts
		self.x = None
		self.u = None
		self.xu = None
		self.delta_x = None

	def step( self, t, state=None ):
		assert( self.state is not None )
		
		if state is not None:
			self.state = state
			
		self.y = integrate.odeint( derivs, self.state, t )
		self.state = self.y[ -1,: ]
		return self.y

	def get_points( self, points ):
		assert( points.shape == ( self.y.shape[0], 4 ) )
		
		points[ :, 0 ] = L1 * sin( self.y[ :,0 ] )
		points[ :, 1 ] = -L1 * cos( self.y[ :,0 ] )

		points[ :, 2 ] = L2 * sin( self.y[ :,2 ]) + points[ :,0 ]
		points[ :, 3 ] = -L2 * cos( self.y[ :,2 ]) + points[ :,1 ]

		self.points = points

		if self.x is None:
			self.x = np.zeros( ( points.shape[0]-1, 4 ) )
		self.x[ :, : ] = points[ :-1, : ]
	
		if self.u is None:
			self.u = np.zeros( ( points.shape[0] - 1, 1 ) )
			self.xu = np.zeros( ( points.shape[0] - 1, 5 ) )
		self.u[ :, 0 ] = self.y[ :-1, 1 ]
		self.xu[ :, 0:4 ] = self.x
		self.xu[ :, 4 ] = self.u[ :, 0 ]
		
		if self.delta_x is None:
			self.delta_x = np.zeros( ( points.shape[0] - 1, 4 ) )
		self.delta_x[ : , : ] = points[ 1: , : ] - points[ :-1, : ]

		return points


	def set_w1( self, w1 ):
		self.state[ 1 ] = w1
		
	def get_x( self ):
		return self.points[ :-1, ]
	
	def get_u( self ):
		return self.y[ :-1, 1 ]
		
	def get_xu( self ):
		return np.hstack( (self.points[ :-1, ], self.y[ :-1, 1 ] ) )
	
	def get_delta_x( self ):
		return self.points[ 1:, ] - self.points[ :-1, ]
