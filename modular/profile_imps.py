from iMPO import *
from iMPS import *
import cProfile
import pstats

cProfile.run("Evec,currVec,densVec = kernel(hamType='tasep',hamParams=(np.random.rand(),np.random.rand(),-np.random.rand()),maxBondDim=100,minIter=20,maxIter=21)",'imps_stats')
p = pstats.Stats('imps_stats')
p.sort_stats('cumulative').print_stats(20)
