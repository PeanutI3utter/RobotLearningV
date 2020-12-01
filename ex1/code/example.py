from jointCtlComp import *
from taskCtlComp import *

select = ['JacTrans', 'JacPseudo', 'JacDPseudo', 'JacNullSpace']
select1 = ['P', 'PD', 'PID', 'PD_Grav', 'ModelBased']

# Controller in the joint space. The robot has to reach a fixed position.
#jointCtlComp([select1[4]], False)

# Same controller, but this time the robot has to follow a fixed trajectory.
#jointCtlComp(['ModelBased'], False)

# Controller in the task space.
taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, pi]).T)

input('Press Enter to close')
