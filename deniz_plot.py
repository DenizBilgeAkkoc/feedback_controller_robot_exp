import torch
import rospy
from nn_models import GeneralModel
from std_msgs.msg import String
import socket
import numpy as np
import matplotlib.pyplot as plt
import math

dataset=np.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/data/torobo/815_trajs_static/trajectories_normalized_34.npy',allow_pickle=True, encoding='latin1')

print(dataset.shape)
# Load the model weights
#rospy.init_node("deniz", anonymous=True)
model = GeneralModel(14,3,7,False)
model.load_state_dict(torch.load('/home/deniz/catkin_ws/src/feedback_controller/fbc/neural_network/weights/815_trajs_static|mse_los|tar_cart|839.189K_params/train_no_0/fbc_1850.pth'))
model.eval()  # Set the model to evaluation mode

#comm=Comm()
elem=dataset[1]
input_s=torch.tensor(elem[0][1:15].tolist())
mean=[ 1.32992016e+00,  1.16893094e+00,  4.75061350e-01,  9.34866042e-01,
 -1.48564714e+00, -1.06043919e+00,  1.07591402e+00, -4.30997512e-03,
 -5.97830623e-06, -7.97757909e-04,  3.14511202e-03, -1.27610726e-02,
  2.15895055e-03,  1.27564559e-02]
mean_tenosr=torch.tensor(mean)

std=[0.18929656, 0.26122264, 0.1385783,  0.48794231, 0.37383509, 0.30146034,
 0.37454724, 0.01025512, 0.0138198,  0.00897769, 0.00892031, 0.02141356,
 0.01329758, 0.02272061]

std_tensor=torch.tensor(std)
input_non=(input_s*std_tensor)+mean_tenosr


#comm.create_and_pub_msg(start_state)
#rospy.sleep(5)

velocities=elem[0][8:15].tolist()
velocities_tensor=input_non[7:]
goal=elem[0][22:25].tolist()
goal_tensor=torch.tensor(goal)


joint_angles_tensor=input_non[:7]

x=[i for i in range(50)]
y1_model=[]
y2_model=[]
y3_model=[]
y4_model=[]
y5_model=[]
y6_model=[]
y7_model=[]

y1_real=[]
y2_real=[]
y3_real=[]
y4_real=[]
y5_real=[]
y6_real=[]
y7_real=[]

for i in range(50):
    input_tensor=(torch.cat((joint_angles_tensor,velocities_tensor),dim=0)-mean_tenosr)/std_tensor
    action=model(goal_tensor,input_tensor)
    print(action[0])
    velocities_tensor=torch.div(action[0],10)
    new_angles=joint_angles_tensor+velocities_tensor
    y1_model.append(float(velocities_tensor[0])* (180.0 / math.pi))
    y2_model.append(float(velocities_tensor[1])* (180.0 / math.pi))
    y3_model.append(float(velocities_tensor[2])* (180.0 / math.pi))
    y4_model.append(float(velocities_tensor[3])* (180.0 / math.pi))
    y5_model.append(float(velocities_tensor[4])* (180.0 / math.pi))
    y6_model.append(float(velocities_tensor[5])* (180.0 / math.pi))
    y7_model.append(float(velocities_tensor[6])* (180.0 / math.pi))


    #comm.create_and_pub_msg(new_angles)
    #rospy.sleep(1)
    joint_angles_tensor=new_angles
#plt.figure(figsize=(12, 6))  # Optional: set figure size
input_s=torch.tensor(elem[1][1:15].tolist())
input_non=(input_s*std_tensor)+mean_tenosr
state_tensor=input_non[:7]

for j in elem[0:]:
    delta_pos=j[15:22]
    print(state_tensor)
    delta_pos_tensor=torch.tensor(delta_pos)
    state_tensor=state_tensor+delta_pos_tensor
    y1_real.append((delta_pos[0])* (180.0 / math.pi))
    y2_real.append((delta_pos[1])* (180.0 / math.pi))
    y3_real.append((delta_pos[2])* (180.0 / math.pi))
    y4_real.append((delta_pos[3])* (180.0 / math.pi))
    y5_real.append((delta_pos[4])* (180.0 / math.pi))
    y6_real.append((delta_pos[5])* (180.0 / math.pi))
    y7_real.append((delta_pos[6])* (180.0 / math.pi))

plt.figure(figsize=(8, 6))
plt.scatter(x, y1_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y1_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 1')
plt.legend()
plt.grid(True)
plt.show()

# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y2_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y2_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 2')
plt.legend()
plt.grid(True)
plt.show()


# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y3_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y3_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 3')
plt.legend()
plt.grid(True)
plt.show()


# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y4_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y4_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 4')
plt.legend()
plt.grid(True)
plt.show()


# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y5_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y5_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 5')
plt.legend()
plt.grid(True)
plt.show()


# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y6_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y6_real, color='blue', label='Data points 1')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 6')
plt.legend()
plt.grid(True)
plt.show()


# Creating the second plot (subplot 2)
plt.figure(figsize=(8, 6))
plt.scatter(x, y7_model, color='red', label='model')  # Scatter plot
plt.scatter(x, y7_real, color='blue', label='ground truth')  # Scatter plot

plt.xlabel('Step number')
plt.ylabel('Delta joint angle')
plt.title('path 0 plot 7')
plt.legend()
plt.grid(True)


# Display the plots
plt.show()

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

