For training I used a Fabric Slice with 1 A30 GPU, 24 processors, 128 GB of RAM, and 200 GB of disc space
The slice has python 3.8.10 installed 


# Steps taken to run on Fabric Slice

## Login to slice:
1. ssh into slice </br>
2. git clone </br>
3. chmod +x fabric_setup.sh </br>
4. ./fabric_setup.sh </br>
5. run data_pipeline, train_cnn and train_fcnn using python3 </br>
