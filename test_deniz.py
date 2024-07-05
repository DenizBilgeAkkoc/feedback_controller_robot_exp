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
model.load_state_dict(torch.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/weights/815_trajs_static|cus_los_5_10_1|tar_cart|45.333K_params/train_no_2/fbc_1100.pth'))
model.eval()  # Set the model to evaluation mode

comm=Comm()
start_state=[0.4793, 0.7613, 0.5457, 1.5242, -2.6195, -0.8813, 2.0450]

comm.create_and_pub_msg(start_state)
rospy.sleep(5)

velocities=[0, 0, 0, 0, 0, 0, 0]
velocities_tensor=torch.tensor(velocities)
goal=[0.5, -0.18, 1.3]
goal_tensor=torch.tensor(goal)
mean=[1.13780162,  0.9468803,   0.94198007,  1.62104903, -1.54787136, -0.45531439,
  0.95075729,  0.01752653,  0.0045933,   0.00977496,  0.00238887,  0.02761652,
  0.01051367, -0.02841756]
mean_tenosr=torch.tensor(mean)
std=[0.46489467, 0.24648595, 0.23237633, 0.30904983, 0.43876924, 0.27400768,
 0.47494448, 0.02223757, 0.0112401,  0.01540193 ,0.01265221, 0.03197541,
 0.0171871,  0.03240877]
std_tensor=torch.tensor(std)
joint_angles_tensor=torch.tensor(start_state)

for i in range(75):
    input_tensor=(torch.cat((joint_angles_tensor,velocities_tensor),dim=0)-mean_tenosr)/std_tensor
    action=model(goal_tensor,input_tensor)
    print(action[0])
    velocities=torch.div(action[0],10)
    new_angles=joint_angles_tensor+velocities
    comm.create_and_pub_msg(new_angles)
    rospy.sleep(0.1)
    joint_angles_tensor=new_angles


# for i in dataset[814]:
#     state_tensor=torch.tensor(comm.joint_state)
#     delta_pos=i[15:22]
#     delta_pos_tensor=torch.tensor(delta_pos)*std_tensor+mean_tenosr
#     state_tensor=state_tensor+delta_pos_tensor
#     comm.create_and_pub_msg(state_tensor)
#     print(delta_pos)
#     print(delta_pos_tensor)
#     print(state_tensor)
#     rospy.sleep(0.1)


# for j in dataset[813:]:
#     comm.create_and_pub_msg(start_state)
#     rospy.sleep(5)
#     for i in j:
#         state_tensor=torch.tensor(comm.joint_state)
#         delta_pos=i[15:22]
#         delta_pos_tensor=torch.tensor(delta_pos)*std_tensor+mean_tenosr
#         state_tensor=state_tensor+delta_pos_tensor
#         comm.create_and_pub_msg(state_tensor)
#         print(delta_pos)
#         print(delta_pos_tensor)
#         print(state_tensor)
#         rospy.sleep(0.1)

