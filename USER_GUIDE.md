# User Guide

#### Preface 

Van Johansen 
Object Detection Capstone Project
GCU 2021

#### Table of contents 

  [General Information](#General-Information)

  [System Summary](#System-Summary)

  [Using the System](#Using-the-System)

  [Troubleshooting](#Troubleshooting)

  [FAQ](#FAQ)

  [Help](#Help)
  
  [Glossary](#Glossary)


![example](https://github.com/vandalismjo/Object_Detection/blob/main/output/output_frame_person/bb_img_164.jpg)

#### General Information 

    The high level steps of the project repository is to read in data and weights, initialize model, import labels, add bounding boxes to images, compile new frames to video, output scores, generate model score, plot model metrics. 
    
    YOLO v3 is using a new network to perform feature extraction which is undeniably larger compare to YOLO v2. This network is known as Darknet-53 as the whole network composes of 53 convolutional layers with shortcut connections (Redmon & Farhadi, 2018). 

    Additional documentation and information can be found in the mAP directory USER GUIDE and the live_pipeline directory USER GUIDE.

#### System Summary 

  The format, and summary, of the Yolo3.main.py script is as follows: 

    - Import all necessary libraries and dependencies. 

    - WeightReader class is used to parse the “yolov3.weights” file and load the model weights into memory. 

    - Set up 53 convolutional layers with shortcut connections. 

    - The _conv_block function is used to construct a convolutional layer 

    - The make_yolov3_model function is used to create layers of convolutional and stack together as a whole. 

    - Define the YOLO v3 model 

    - Load the pre-trained weights  

    - Save the model using Keras save function and specifying the filename 

    - The decode_netout function is used to decode the prediction output into boxes 

    - Scale and stretch the decoded boxes to be fit into the original image shape 

    - The bbox_iou function is used to calculate the IOU (Intersection over Union) by getting the _interval_overlap of two boxes 

    - The do_nms function is used to perform NMS 

      - NMS is performed as follows: 

        - Select the box that has the highest score. 

        - Calculate the interval overlap of this box with other boxes, and omit the boxes that overlap significantly (iou >= iou_threshold). 

        - Repeat to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.  

    - The get_boxes function is used to obtain the boxes which have been selected through NMS filter 

    - The draw_boxes function is used to draw a rectangle box to the input image using matplotlib.patches.Rectangle class 

    - Declare several configurations: 

      - Anchors: carefully chosen based on an analysis of the size of objects in the COCO dataset. 

      - Class_threshold: the probability threshold for detected objects 

      - Labels: class labels from the COCO dataset 

    - Iterate over input frames to make predictions. 

#### Using the System 

    The repository will generate all needed results by running the single main jupyter notebook in the root directory. There are several configurations that can be modified like file paths inside the ‘main_config.yaml’ file. The project will output model results through the main notebook. If scoring metrics and plots and visualizations around the scoring are needed, they can be run through the scoring pipeline. The scoring pipeline is inside the mAP directory and all output can be generated from the main.py script. In order to run the live object detection, end users will need to change directory to the 'live_pipeline' and from there run the following two commands in the prompt: 
      - python load_weights.py  
      - python detect_video.py --video 0 

    There are several plots that get generated and saved, the first comes from the main notebook in the root directory. This line plot counts the number of detected people in frame. This use case can be used for security managers who only require x number of people in a single location at once. This plot can also be turned into an alarm to notify end users when the detected number of people in frame exceeds a specified number for a certain amount of time. Other plots are generated in the scoring pipeline, of which plot the mean average precision, mAP, of the model. These scores are divided into classes and plotted against each other. For example, if there are 2 different objects in the video, people and cars, the model will score and plot how well it classified people compared to how well it classified cars. This is great to inform end users and developers of any modifications are needed in the training process or input data when using the product.

#### Troubleshooting 

    - Issues may arise if file names and paths are not unified and set properly in the main configuration file.
    - Live object detection from web camera must be called from the terminal in the directory of 'live_pipeline'

#### FAQ 

    How do I turn a video into frames to run through the detection pipeline?
     - Inside the yolo3 tools path there is a python script that uses OpenCV to take the new video and compile it into several frames.
    
    How do I turn my detected frames with bounding boxes into a video?
     - Inside the yolo3 tools path there is a python script that uses OpenCV to take the new frames and compile it into a video.
    
    How do I generate metrics on new input?
     - Ground truth information is required. If no ground truth information is provided or available, that will need to be generated outside of this product. Ground truth files will need to fit the standard output as exampled in current repository.

#### Help

	For any additional help, questions, or resources please contact the following email: 
    - VJohansen@my.gcu.edu 

#### Glossary

  Bounding Box 
    - The resulting output, normally a box, showing the model's prediction on detected objects. 

  COCO 
    - Common Object in COntext - a large-scale object detection, segmentation, and captioning dataset. 

  Configuration File
    - A YAML file that contains system configurations, usually the source of truth for hard coded values. 

  Darknet 
    - An open-source neural network framework written in C and CUDA. 

  False Negative 
    - An outcome where the model incorrectly predicts the negative class. 

  False Positive 
    - An outcome where the model incorrectly predicts the positive class. 

  GitHub 
    - A provider of Internet hosting for software development and version control using Git. It offers the distributed version control and source code management functionality of Git, plus its own features. 

  Ground Truth 
    - User defined coordinates for known objects in frame, used for scoring model predictions against. 

  IOU 
    - Intersection Over Union - computes intersection over the union of the bounding box for the ground truth and the predicted bounding box. 

  Jupyter Notebook 
    - An open-source web application that allows users to create and share documents that contain live code, equations, and visualizations. 

  NMS 
    - Non-Maximal Suppression - a class of algorithms to select one entity (e.g., bounding boxes) out of many overlapping entities. 

  Pipeline 
    - The set of processes that convert raw data into actionable answers to business questions. 

  Precision 
    - Also known as positive predictive value, is the fraction of relevant instances among the retrieved instances. 

  Recall 
    - Also known as sensitivity, is the fraction of relevant instances that were retrieved. 

  Repository 
    - A place that hosts an application's code source, together with various metadata. 

  Sensitivity 
    - The fraction of relevant instances that were retrieved. 

  TensorFlow 
    - An end-to-end open-source platform for machine learning that has a comprehensive, flexible ecosystem of tools and libraries. 

  Terminal 
    - A textual way to interact with the operating system, a place where python interpreter can be called and run python scripts. 

  True Negative 
    - An outcome where the model correctly predicts the negative class. 

  True Positive 
    - An outcome where the model correctly predicts the positive class. 

  Visual Studio Code 
    - A source-code editor made by Microsoft for Windows, Linux and macOS. 

  YOLO 
    - You Only Look Once - A state-of-the-art, real-time object detection system. 

