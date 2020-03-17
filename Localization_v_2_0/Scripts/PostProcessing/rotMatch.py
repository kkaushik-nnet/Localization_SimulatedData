import numpy as np

def ptSSE( pt, X ):
    '''
    Point-wise smallest squared error.
    This is the distance from the point `pt`
    to the closest point in `X`
    '''
    difference = pt - X
    # x and y columns
    xcol = np.ravel( difference[:,0] )
    ycol = np.ravel( difference[:,1] )
    # sum of the squared differences btwn `pt` and `X`
    sqr_difference = (xcol**2.0 + ycol**2.0)
    # nearest squared distance
    distance = np.min( sqr_difference )
    # index of the nearest point to `pt` in `X`
    nearest_pt = np.argmin( sqr_difference )
    return distance
 
def NSSE( X, Y ):
    '''
    Nearest sum squared error.
    This is the sum of the squares of the
    nearest differences between the points
    of `X` and the points of `Y`
    '''
    err = 0.0
    for x in X:
        err += ptSSE( x, Y )
    return err


def _fit( X, Y, subsample=10, nrot=100, minRot=0., maxRot = 2*np.pi):
   rotMat = np.matrix(np.zeros([2,2]))
   bestMatch = None
   bestPhi   = None
   bestY     = None

   for phi in np.linspace(minRot, maxRot, nrot):
       rotMat[0,0] = rotMat[1,1] = np.cos(phi)
       rotMat[1,0] = -np.sin(phi)
       rotMat[0,1] =  np.sin(phi)
       Yout = Y * rotMat
       thisNSSE = NSSE(X[::subsample,:], Yout)
       if bestMatch is None or bestMatch>thisNSSE:
          bestMatch = thisNSSE
          bestPhi = phi
          bestY = Yout.copy()
   return bestY, bestPhi, bestMatch

def fit( X, Y):
   # find best rotation 
   bestY, bestPhi, bestMatch = _fit( X, Y, subsample=5, nrot=100)
   print "best approximate map rotation phi=%.2f, err=%.2f"%( bestPhi, bestMatch)
   bestY, bestPhi, bestMatch = _fit( X, Y, minRot=bestPhi-0.05,  maxRot=bestPhi+0.05, nrot=100)
   print "      best exact map rotation phi=%.2f, err=%.2f"%(bestPhi, bestMatch)
   return bestY, bestPhi, bestMatch
