import matplotlib.pyplot as plt
import numpy



def moving_average(interval, window_size):
    window = numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

window_size = 1 # Plot running average

#c_data = numpy.genfromtxt("rew_cacc_1.csv", delimiter=",")
#data = numpy.genfromtxt("rew_ours_1.csv", delimiter=",")
#h_data = numpy.genfromtxt("rew_human.csv", delimiter=",")

#file_list = [["window_1_run.txt","r"],["window_5_run.txt","g"],["window_10_run.txt","b"]]

#file_list = [["window_5_run.txt","g"],["cacc_start.txt","b"]]

# Filename
file_list = [["output_good.log","g"]]

for file_name in file_list:

        train_data = []
        train_data_se = []

        #f=open("run_4.txt","r")
        f=open(file_name[0],"r")
        contents = f.read()

        for line in contents.split("\n"):
                if "Long Eval: Average Score:" in line:
                        print(line)
                        train_data.append(float(line.replace("Long Eval: Average Score: ","").split("SE Score: ")[0].replace(" ","")))
                        train_data_se.append(float(line.replace("Long Eval: Average Score: ","").split("SE Score: ")[1].replace(" ","")))

        m_train_data = []

        #m_f=open("run_7.txt","r")
        m_f=open(file_name[0],"r")
        m_contents = m_f.read()

        #for line in m_contents.split("\n"):
                #if "Long Eval: Median Score:" in line:
                        #m_train_data.append(float(line.replace("Long Eval: Median Score:","").replace(" ","")))

        plt.plot(moving_average(train_data,window_size)[window_size:-window_size],file_name[1])
        #plt.plot(moving_average(m_train_data,window_size)[window_size:-window_size] * 55,color=file_name[1],linestyle="--")

# Compare with Human

plt.title("Learning Curve")
# Human and CACC raw
plt.axhline(y=110, color='b', linestyle='-')
plt.axhline(y=50, color='r', linestyle='-')
#plt.plot(moving_average(c_data,window_size)[window_size:-window_size], color='b')
#plt.plot(moving_average(data,window_size)[window_size:-window_size],color="r")
#plt.plot(moving_average(c_data,window_size)[window_size:-window_size][:len(train_data)], color='b')
#plt.plot(moving_average(data,window_size)[window_size:-window_size][:len(train_data)],color="r")
#plt.plot(moving_average(h_data,window_size)[window_size:-window_size],color="g")

#train_data = moving_average(train_data,window_size)[window_size:-window_size]
#train_data_se = moving_average(train_data_se,window_size)[window_size:-window_size]
#plt.fill_between(range(len(train_data)),numpy.array(train_data)-numpy.array(train_data_se), numpy.array(train_data)+numpy.array(train_data_se),color="r")
plt.ylabel("Reward")
plt.xlabel("Epoch")
plt.show()