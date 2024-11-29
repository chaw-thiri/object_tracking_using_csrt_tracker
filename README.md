# What is CSRT? How does CSRT work and why is it better than other OpenCV built-in tracker?
[See the ppt](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/CSRT_tracker.pptx)
# How was it modifiy to get better results
There are multiple approaches available to modify the built-in API to be well-adjusted for multiple situations. Here I have demonstrated a couple of them (except the number 2) until the results satisfy me.
1) [Default API only](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/other_attempts/api_only.py)
2) Default API + Faster RCNN ( not Included ) 
3) [Default API + YOLOv8](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/other_attempts/api_yolo.py)
4) [Default Model + Custom model](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/other_attempts/api_%26_custom_model.py)
5) [Default Model + Custom model + Image Preprocessing](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/other_attempts/defaultAPI_customModel_preprocessing.py)
6) [FINAL MODEL --- Default Model + Image Preprocessing](https://github.com/chaw-thiri/object_tracking_using_csrt_tracker/blob/main/main.py)
# Test videos with different purposes ( [Downloadable sources](https://drive.google.com/drive/folders/1VKnjS3lqOyAvjof0VGzyystAsdQhkN6K?usp=sharing)) 
The strength of the API lies in being able to **handle objects in scale variations, occlusion and complex backgrounds**. I have selectively choosen the test videos that can demonstrate those capacities.  

1) people.mp4 >> small object detection
2) market.mp4 >> occlusion + clutter background
3) race_car.mp4 >> perspective change + rapid motion
4) boat.mp4 >> scaling
5) smallest.mp4 >> scaling ( optimized by image preprocessing )
6) rabbit.MOV >> Similarity with the background , occlusion ( can only be detected by the final model )
# Record of tested videos 
[Here](https://drive.google.com/drive/folders/1COT3M-OI-PS3Zf5FcWdbmOr3M1sBh-cD?usp=sharing) 

# Discussion 
* The built-in API performs great in most situation, however I failed to track my bunny in self-made rabbit.MOV which leads me to research into possbile upgrades.   
* Most commonly researched method for CSRT tracker combination is the **default tracker + Faster RCNN**. However, that is left out in this project due to the heaviness of the model and training requirement.
***Coupling the tracker with pre-trained YOLOv8 model** showed no signigicant improvement ( This couple did not pass the rabbit.MOV and oeople.mp4 videos) even though it slowed down the frame rate.
* Using **custom model but adjusting the parameters**: shows dis-satisfactory results. In my model, even though it can detect the rabbit.MOV well, it failed to track the phone in the people.mp4 and got fixed on the background scene.
***Swapping between the default model and custom model:** does not perform as expected as well as the main problem is the model not recognizing the object is lost.
* ### Image pre-processing techinques : Histogram normalization and Edge detection
* Histogram normalization enhances contrast and reduces sensitivity to lighting variations, ensuring consistent features for the CSRT tracker.
* Edge detection highlights object boundaries, improving initial target definition and distinguishing the object from background clutter.
* Together, these preprocessing steps improve the tracker's accuracy, reduce drift, and ensure better feature extraction, enabling more robust and reliable object tracking.

