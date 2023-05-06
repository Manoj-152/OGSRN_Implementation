# SAR-Super-resolution-v1

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

Before running the code, the directory to input SAR images must be given (Inference Images by default). Any number of SAR images can be put in the folder and the model would be able to generate super-resolved SAR images for each of them. Note that the SAR images must be in tif format.

## Test_Sentinel jupyter notebook
This notebook contains the same codes which are used for inference in the script. It provides a visual understanding of what is done at each step (from preprocessing to super-resolving the image).

## Solution Development
* The model is trained on 18717 SAR-optic pairs of patches. Patch size is 256x256
* Model: SORTN (Present in next branch) is the generator using to obtain optic image from SAR
* SRUN: Responsible for super-resolving the image, architecture similar to U-Net
* Loss Functions: Content Loss + 0.1 x Evaluation Loss is used for SRUN. cGAN loss + 100 * L1 loss between optical ground truth and optical generated is used for SORTN.
* For inference, preprocessing involving clipping, despeckling, adaptive histogram scaling, minmax scaling between -1 to 1 and cropping to required patches is done. The patches are super-resolved and later stitched together. The steps can be visualized in the Test_Sentinel jupyter notebook.

## Results
(a.) Spacenet 6 dataset
![Screen Shot 1944-02-23 at 3 05 57 PM](https://user-images.githubusercontent.com/82506345/168256472-910eadd5-8345-4a6c-8bb4-84dfb5758c45.png)



(b.) Sentinel -1 

![imgonline-com-ua-twotoone-RgxpMML7YNeRf](https://user-images.githubusercontent.com/82506345/168259244-e30333f6-6dff-4788-891d-23eff516af76.jpeg)


