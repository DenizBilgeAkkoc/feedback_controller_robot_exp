import torch
import rospy
from nn_models import GeneralModel
from torobo_msgs.msg import ToroboJointState
from std_msgs.msg import String
import socket
import numpy as np

class Comm():
    def __init__(self):
        self.joint_state=None
        self.ezcom='/torobo/ezcommand'
        self.state_topic='/torobo/right_arm_controller/torobo_joint_state'

        rospy.Subscriber(self.state_topic,ToroboJointState, callback=self.call_func)
        self.pub=rospy.Publisher(self.ezcom, String, queue_size=1)

    def call_func(self, data):
        self.joint_state=data.position

         
    def create_and_pub_msg(self,positions):

        localIP     = "192.168.5.119"
        localPort   = 20001

        targetIP    = "192.168.5.112"
        targetPort  = 50000

        numBytes2Get = 1024

        # Create a UDP socket. UDP is datagram based.
        server         = (targetIP, targetPort)
        udpClientSocket     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if type(positions)!=list:
            positions=positions.tolist()
        command='setq rarm'
        for i in positions:
                command=command + ' ' + str(i)

        encmess = command.encode()

        sentBytesCount      = udpClientSocket.sendto(encmess, server)
        print ('>',sentBytesCount,'bytes are/is just sent!')

dataset=np.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/data/torobo/815_trajs_static/trajectories_normalized.npy',allow_pickle=True, encoding='latin1')


# Load the model weights
rospy.init_node("deniz", anonymous=True)
model = GeneralModel(14,3,7,False)
model.load_state_dict(torch.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/weights/815_trajs_static|mse_los|tar_cart|839.189K_params/train_no_0/fbc_2100.pth'))
model.eval()  # Set the model to evaluation mode

comm=Comm()
start_state=[1.5423, 0.9664, 0.8408, 1.8099, -1.1050, -0.1675, 0.7178]
comm.create_and_pub_msg(start_state)
rospy.sleep(5)



velocities=[0, 0, 0, 0, 0, 0, 0]
velocities_tensor=torch.tensor(velocities)
goal=[0.5, -0.1, 1.15]
goal_tensor=torch.tensor(goal)
mean=[9.41301546e-01,  9.02411348e-01,  5.21996750e-01,  1.73236606e+00,
 -1.85313919e+00, -5.38227982e-01,  1.43407658e+00, -5.51738653e-03,
 -7.88986099e-03,  3.93582725e-03,  8.11379711e-03, -1.02461122e-03,
 -4.51122273e-03, -1.45835481e-03]
mean_tenosr=torch.tensor(mean)

std=[0.32050168, 0.376122,   0.33323269, 0.21498294, 0.470825,   0.32965009,
 0.41975237, 0.01382645, 0.01845979, 0.01897269, 0.01006318, 0.01948571,
 0.01880103, 0.01840387]

std_tensor=torch.tensor(std)
joint_angles_tensor=torch.tensor(start_state)

for i in range(50):
    input_tensor=(torch.cat((joint_angles_tensor,velocities_tensor),dim=0)-mean_tenosr)/std_tensor
    action=model(goal_tensor,input_tensor)
    velocities_tensor=torch.div(action[0],10)
    print(velocities_tensor)
    new_angles=joint_angles_tensor+velocities_tensor
    comm.create_and_pub_msg(new_angles)
    rospy.sleep(0.2)
    joint_angles_tensor=new_angles

# for i in dataset[0]:
#     state_tensor=torch.tensor(comm.joint_state)
#     delta_pos=i[15:22]
    
#     delta_pos_tensor=torch.tensor(delta_pos)
#     delta_pos_tensor=torch.mul(delta_pos_tensor,10)
#     state_tensor=state_tensor+delta_pos_tensor
#     comm.create_and_pub_msg(state_tensor)
#     print(delta_pos)
#     print(delta_pos_tensor)
#     print(state_tensor)
#     rospy.sleep(0.1)
