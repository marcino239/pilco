import numpy as np
import scipy as sp

from pendubot import Pendubot
from numpy import pi
from sklearn import gaussian_process
from scipy.special import erf
from GPS import GPS

def get_policy_params( fi ):
	'''
		returns (A,b)
	'''
	return fi[ 0:4 ], fi[ 4 ]

# calculate policy parameters
def cost_t( x, target_point ):
	return np.exp( -np.sum( (x-target_point)**2 ) )


def cost( fi, gp, target_point, pbot, vx0 ):
	'''
		calculates cost for a given policy fi
		based on dynamics model gp
		and trajectory of system variables x
	'''
	A, b = get_policy_params( fi )

	temp_c = numpy.zeros( points_x.shape[ 0 ] )

	for i in range( points_x.shape[ 0 ] ):
		mxt_1 = np.atleast_2d( np.avg( points_x[ :i, : ] ) )
		if i < MIN_X_SAMPLES:
			vx_1 = vx0
		else:
			vx_1 = np.cov( points_x[ :i, : ] )

		mxu = np.row_stack( (mx_1, A * mx_1 + b) )
		vxu = np.matrix( [ [ vx_1, vx_1 * A.T ], [ A * vx_1, A * vx_1 * A.T ] ] )

		xu = np.random.multivariate_normal( mxu, vxu, NUM_XU_SAMPLES )

		# apply dynamics
		delta = gp.predict( xu )

		# recover xt
		xt_1, ut_1 = xu
		xt = xt_1 + delta

		# integrate cost
		temp_c[ i ] = np.apply_along_axis( cos_t, axis=1, arr=xt, args=target_point ).sum()

	return temp_c.sum()


def run_on_robot( pbot, A, b, max_time, dt_pbot, dt_pilco ):
	'''
		function runs pbot with a controller (A,b) for max_time
	'''
	
	num_samples = int( max_time / dt_pilco )
	
	x = np.zeros( (num_samples+2, 4) )
	u = np.zeros( (num_samples+2, 1) )
	
	t = np.linspace( 0.0, dt_pilco / dt_pbot, dt_pbot )
	points = np.zeros( ( t.shape[0], 4 ) )

	pbot.step( t )
	pbot.get_points( points )

	for i in range( int( max_time / dt_pbot ) ):
		if i == 0:
			x[ 0, : ] = pbot.x[ 0, : ]
			u[ 0, : ] = pbot.u[ 0, : ]
			x[ 1, : ] = pbot.x[ -1, : ]
			u[ 1, : ] = pbot.u[ -1, : ]
		elif i % int( dt_pilco / dt_pbot ) == 0:
			x[ i + 1, : ] = pbot.x[ -1, : ]
			u[ i + 1, : ] = pbot.u[ -1, : ]
			
		pbot.step( t )
		pbot.get_points( points )

	return x, u


# init
gps = GPS( n_outputs = 4, regr = 'constant', theta0 = 100. )

# initialise pendubot
start_state = np.array( [ pi/4., 0., 0., 0. ] )
target_point = np.array( [0., 1., 0., 2.] )		# observing (x,y) coordinates of 2 ends

# init policy parameters
fi = np.random.normal( 0.0, 1.0, size=5 )

# sampling period for pbot
dt_pbot = 0.010
t_full = np.arange( 0.0, 40., dt_pbot )
points = np.zeros( ( t_full.shape[0], 4 ) )

# sampling period for PILCO
dt_pilco = 1./20.

pbot = Pendubot( start_state )

# PILCO algo:
# get some random moves ( or initial training )
pbot.step( t_full )
pbot.get_points( points )

vx0 = np.eye( target_point.shape[0] ) * 0.01

epoch = 0
loop_flag = True
while loop_flag:
	print( 'epoch: {0}'.format( epoch ) )
	# fit dynamics model: (xt-1,ut-1) -> delta t
	print( '\tfitting dynamics' )
	gps.fit( pbot.xu, pbot.delta_x )

	# minimise cost given policy
	print( '\toptimising policy' )
	args = ( gps, target_point, pbot.get_x(), vx0 )

	res = sp.optimize.minimize( cost, fi, args, method='bfgs', jac=False )
	fi = res.x
	A, b = get_policy_params( fi )

	# run on robot, records points
	pbot = Pendubot( start_state )
	points_x, points_u = run_on_robot( pbot, A, b )
	
	# check if we achived required precision
	loop_flag = abs( cost_t( points_x[ -1, : ], target_point ) ) < X_PRECISION 
	epoch += 1
