import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pendubot import Pendubot
from numpy import pi

# set initial state
start_state = np.array( [ pi/2.0, 0.0, 0.0, 0.0 ] )
pbot = Pendubot( start_state )

# create a time array from 0..100 sampled at 0.05 second steps
dt = 0.05
t = np.arange( 0.0, 1, dt )

# holds evolved points
points = np.zeros( ( t.shape[0], 4 ) )

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot( [], [], 'o-', lw=2 )
time_template = 'time = %.1fs'
time_text = ax.text( 0.05, 0.9, '', transform=ax.transAxes )

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate( i ):
	
	k = i % t.shape[ 0 ]
	if k == 0:
		pbot.step( t )
		pbot.get_points( points )
	
	thisx = [ 0, points[ k, 0 ], points[ k, 2 ] ]
	thisy = [ 0, points[ k, 1 ], points[ k, 3 ] ]

	line.set_data( thisx, thisy )
	time_text.set_text( time_template % (i*dt) )
	return line, time_text

ani = animation.FuncAnimation( fig, animate, interval=int(dt * 1000.0), blit=True, init_func=init)

#ani.save('double_pendulum.mp4', fps=15)
plt.show()
