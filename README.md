# SORTN (SAR-to-Optical Residual Translation Network)

Weights for SORTN model (required to run the code) can be downloaded from the following S3 URI link. 

s3://galaxeye-bucket/SAR-SR/SORTN_best_ckpt.pth

## Train
Command: 

pip3 install -r requirements.txt

./run_script.sh train

Installs the required libraries from requirements.txt and runs the train script.

Before running the code, respective directories for the dataset and the directories where the results are to be stored must be given in the config.yaml file. New folders will be created for result directories already if they are not present.

## Test/Inference
Command:

pip3 install -r requirements.txt

./run_script.sh inference

Before running the code, the path to the input SAR must be changed in the config file.
