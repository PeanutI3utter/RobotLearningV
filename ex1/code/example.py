from jointCtlComp import *
from taskCtlComp import *

select = ['JacTrans', 'JacPseudo', 'JacDPseudo', 'JacNullSpace']
# Controller in the joint space. The robot has to reach a fixed position.
#jointCtlComp(['ModelBased'], False)

# Same controller, but this time the robot has to follow a fixed trajectory.
jointCtlComp(['ModelBased'], False)

# Controller in the task space.
#taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, pi]).T)

#input('Press Enter to close')
