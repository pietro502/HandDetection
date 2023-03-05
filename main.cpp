//Written and revised by Paolo Bresolin, Giacomo Gonella and Pietro Picardi

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "libraries/score.h"
#include "libraries/NN.h"

using namespace std;
using namespace cv;
using namespace dnn;

//Main function
int main() {
    //Written by Paolo Bresolin
    //Loads configuration and weight files for the model
    string modelConfiguration = "../NN/yolov3_testing.cfg";
    string modelWeights = "../NN/yolov3_training_last.weights";

    //Written by Paolo Bresolin
    //Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    //Written by Giacomo Gonella
    //Load test images and respective ground truths
    Mat img, true_mask;
    string mask_path = "../test/mask";
    string det_path = "../test/det";
    string rgb_path = "../test/rgb";
    vector<string> filenames_rgb, filenames_det, filenames_mask;
    glob(mask_path, filenames_mask, false);
    glob(rgb_path, filenames_rgb, false);
    glob(det_path, filenames_det, false);
    
    //Written by Paolo Bresolin except for the highlighted lines
    //Process each image at the time
    for(int i = 1; i < filenames_rgb.size(); i++) {

        //Load mask and image, then create blob from image
        true_mask = imread(filenames_mask[i]);
        img = imread(filenames_rgb[i]);
        Mat blob;
        blobFromImage(img, blob, 1 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);

        //Written by Pietro Picardi
        vector<SCORE::Box> true_bb;
        getTrueBoundingBoxes(filenames_det[i], &true_bb, img);

        //Set the blob as input of the network
        net.setInput(blob);

        //Run the forward pass to get the output of the output layers
        vector<Mat> outputs;
        net.forward(outputs, getOutputsNames(net));

        //Process the image considering the extracted bounding boxes
        postprocess(img, outputs, true_bb, true_mask, i);

        //Written by Pietro Picardi
        //Show the image with the detection boxes and segmentation masks
        /*
        imshow(filenames_rgb[i], img);
        waitKey(0);
        destroyAllWindows();
        */
    }
    return 0;
}
