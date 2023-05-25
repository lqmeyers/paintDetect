# paintDetect
### A repo for the development of automatic UNet based segmentation of bee images
Based on milesial/Pytorch-UNet/ 
Used to train models for paint detection and mask generation on training dataset of 192 image/mask combos. 
<p align="left"> Sample prediction: </p>


- *Image name: f17x2022_06_28.mp4.track000206.frame006589.jpg* 
- *Dice coefficient: 0.9417218543046357*
<p align="center"> <img src=https://github.com/lqmeyers/paintDetect/assets/107192889/ab8ebf3e-6554-4e12-8e5d-10017724facb /></p>


<p align="center"> Green = True Mask, Blue = Predicted Mask, Red = Overlap </p>

<p align="left"> Training visualized using the weights and balances library and web api: </p>
<p align="center"><img src=https://github.com/lqmeyers/paintDetect/assets/107192889/1e0c1838-2022-434a-a7fe-715eb1411c46)  width="600" height="300" border="10"/></p>
