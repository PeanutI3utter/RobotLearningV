from jointCtlComp import *
from taskCtlComp import *
from multiprocessing import Process
import time
import matplotlib as plt

select = ['JacTrans', 'JacPseudo', 'JacDPseudo', 'JacNullSpace']
select1 = ['P', 'PD', 'PID', 'PD_Grav', 'ModelBased']

def plot_that(sel):
    taskCtlComp([select[3]], 1e-15, np.mat([0, pi if sel else -pi]).T)
    time.sleep(10000)

if __name__ == '__main__':  
    # Controller in the joint space. The robot has to reach a fixed position.
    jointCtlComp(['PD', 'PD_Grav'], False)

    # Same controller, but this time the robot has to follow a fixed trajectory.
    #jointCtlComp(['ModelBased'], False)

    # Controller in the task space.

    if False:
        p1 = Process(target=plot_that, args=(1,))
        p2 = Process(target=plot_that, args=(1,))

        p1.start()
        p2.start()
        p1.join()
        p2.join()

    input('Press Enter to close')