import numpy as np
from sklearn import gaussian_process
from sklearn.base import BaseEstimator

MACHINE_EPSILON = np.finfo(np.double).eps

class GPS( BaseEstimator ):
	'''
	multivariete wrapper for gaussian process model for sklearn
	'''
	
	
	def __init__( self, n_outputs, regr='constant', corr='squared_exponential',
                 storage_mode='full', verbose=False, theta0=1e-1 ):
		self.gps = [ gaussian_process.GaussianProcess( regr=regr, corr=corr,
                 storage_mode=storage_mode, verbose=verbose, theta0=theta0 ) for i in range( n_outputs ) ]
	
	def fit( self, X, Y ):
		assert( len( self.gps ) == Y.shape[ 1 ] )
		
		for i in range( len( self.gps ) ):
			try:
				self.gps[i].fit( X, Y[ :, i ] )
			except ValueError as e:
				print( 'ValueError cought for i:{0}: e:{1}'.format( i, e ) )
				raise e

		return self.gps
		
	def predict( self, X ):
		n_outputs = len( self.gps )
		Y = np.empty( (X.shape[0], n_outputs) )
		
		for i in range( n_outputs ):
			Y[ :, i ] = self.gps[ i ].predict( X )
		
		return Y

