//Written and revised by Paolo Bresolin, Giacomo Gonella and Pietro Picardi

#include <opencv2/dnn.hpp>
#include "score.h"
#include "RegionGrowing.h"
#include "NN.h"

using namespace cv;
using namespace std;
using namespace dnn;

//Written by Paolo Bresolin
/*
Find the names of the output layers of $net, in order to compute then the output using function $forward.
Return a vector of strings containing the names of the output layers of $net.
*/
vector<string> getOutputsNames(const Net& net) {
    //Vector of names
    vector<string> names;
    //Get the indices of the output layers
    vector<int> outLayers = net.getUnconnectedOutLayers();
    //Get the names of all the layers in the network
    vector<string> layersNames = net.getLayerNames();
    //Get the names of the output layers
    for (int i = 0; i < outLayers.size(); i++)
        names.push_back(layersNames[outLayers[i] - 1]);
    return names;
}//getOutputNames

//Written by Giacomo Gonella and Pietro Picardi
//Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& img, Mat& predicted_mask, const Scalar color) {
    //Written by Pietro Picardi
    double alpha = 0.6;
    Mat subImg, subMask, test;
    //Selecting each bounding-box and its content
    if (0 <= left && 0 <= (right - left) && 0 <= top && 0 <= (bottom - top)) {
        if (right > img.cols && bottom <= img.rows)
            right = img.cols;
        else if (right <= img.cols && bottom > img.rows)
            bottom = img.rows;
        else if (right > img.cols && bottom > img.rows) {
            right = img.cols;
            bottom = img.rows;
        }
    }
    Rect roi(left, top, right - left, bottom - top);
    subImg = img(roi);
    subMask = predicted_mask(roi);
    test = subImg.clone();
    //Using RGBA color space
    Mat rgba;
    cvtColor(subImg, rgba, COLOR_BGR2RGBA);
    //Splitting RGBA channels
    vector<Mat> rgba_planes;
    split(rgba, rgba_planes);
    Mat r = rgba_planes[0];
    Mat g = rgba_planes[1];
    Mat b = rgba_planes[2];
    Mat a = rgba_planes[3];
    //Using YCrCb color space
    Mat yCrCb;
    cvtColor(subImg, yCrCb, COLOR_BGR2YCrCb);
    //Splitting YCrCb channels
    vector<Mat> yCrCb_planes;
    split(yCrCb, yCrCb_planes);
    Mat y = yCrCb_planes[0];
    Mat cr = yCrCb_planes[1];
    Mat cb = yCrCb_planes[2];
    //Using HSV color space
    Mat hsv;
    cvtColor(subImg, hsv, COLOR_BGR2HSV);
    //Splitting YCrCb channels
    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);
    Mat h = hsv_planes[0];
    Mat s = hsv_planes[1];
    Mat v = hsv_planes[2];
    Mat mask = Mat::zeros(subImg.rows, subImg.cols, subImg.type());
    int r_val, b_val, g_val, a_val, y_val, cr_val, cb_val, h_val, s_val, v_val;
    for (int i = 0; i < subImg.rows; i++)
        for (int j = 0; j < subImg.cols; j++) {
            r_val = r.at<uchar>(i, j);
            b_val = b.at<uchar>(i, j);
            g_val = g.at<uchar>(i, j);
            a_val = a.at<uchar>(i, j);
            y_val = y.at<uchar>(i, j);
            cr_val = cr.at<uchar>(i, j);
            cb_val = cb.at<uchar>(i, j);
            h_val = h.at<uchar>(i, j);
            s_val = s.at<uchar>(i, j);
            v_val = v.at<uchar>(i, j);
            if (
                (
                    (r_val > 95) &&
                    (g_val > 40) &&
                    (r_val > g_val) &&
                    (r_val > b_val) &&
                    (abs(r_val - g_val) > 15) &&
                    (a_val > 15)
                    )
                &&
                (
                    (
                        (cr_val > 135) &&
                        (cb_val > 85) &&
                        (y_val > 80) &&
                        (cr_val <= ((1.5862 * cb_val) + 20)) &&
                        (cr_val >= ((0.3448 * cb_val) + 76.2069)) &&
                        (cr_val >= ((-4.5652 * cb_val) + 234.5652)) &&
                        (cr_val <= ((-1.15 * cb_val) + 301.75)) &&
                        (cr_val <= ((-2.2857 * cb_val) + 432.85))
                        )
                    ||
                    (
                        (h_val >= 0) &&
                        (h_val <= 50) &&
                        (s_val >= 23) &&
                        (s_val <= 68)
                        )
                    )
                )
                mask.at<Vec3b>(i, j) = subImg.at<Vec3b>(i, j);
        }

    //Written by Giacomo Gonella
    //Region growing
    /*
    imshow("color space mask", mask);
    waitKey(0);
    destroyAllWindows();
    */
    vector<Point> seeds = selectInitialPoints(mask); //Each point is (row, col)
    Mat seed_mask = getSeedMaskFromPoints(subImg, seeds);
    /*
    imshow("seed mask", seed_mask);
    waitKey(0);
    destroyAllWindows();
    */
    int threshold = 50;
    regionGrowing(subImg, seeds, seed_mask, threshold);
    /*
    imshow("after region growing", seed_mask);
    waitKey(0);
    destroyAllWindows();
    */
    colorBoundingBox(subImg, seed_mask, color);
    /*
    imshow("segmented bb", subImg);
    waitKey(0);
    destroyAllWindows();
    */
    addWeighted(test, alpha, subImg, 1 - alpha, 0, subImg);
}

//Written by Paolo Bresolin, Giacomo Gonella and Pietro Picardi
//Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& img, const vector<Mat>& outputs, const vector<SCORE::Box> true_bb, Mat& true_mask, const int k) {

    //Written by Paolo Bresolin
    //Define some needed variables
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<double> iou_score;
    double pixel_accuracy;
    Mat predicted_mask = Mat::zeros(img.size(), img.type()), img2 = img.clone();
    /*
    Scan all the bounding boxes returned in output from the network and keep only those ones with high a confidence score.
    Assign the box's class label as the class with the highest score for the box.
    */
    for (int i = 0; i < outputs.size(); i++) {
        float* data = (float*)outputs[i].data;
        for (int j = 0; j < outputs[i].rows; ++j, data += outputs[i].cols) {
            Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
            Point classIdPoint;
            double confidence;
            //Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            //Value of the confidence set after some trials
            //The bounding box is saved if and only if ($confidence > 0.65)
            if (confidence > 0.65) {
                int centerX = (int)(data[0] * img.cols);
                int centerY = (int)(data[1] * img.rows);
                int width = (int)(data[2] * img.cols);
                int height = (int)(data[3] * img.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }//if
        }//for
    }//for

    //Written by Pietro Picardi
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    vector<Scalar> colors;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    for (int i = 0; i < indices.size(); i++) {

        int idx = indices[i];
        colors.push_back(Scalar(
            (double)rand() / RAND_MAX * 255,
            (double)rand() / RAND_MAX * 255,
            (double)rand() / RAND_MAX * 255
        ));
        SCORE::Box b = true_bb.at(true_bb.size() - 1 - i), b_predicted;
        Rect box = boxes[idx];
        b_predicted.p1 = SCORE::Vec2<double>(box.x, box.y);
        b_predicted.p2 = SCORE::Vec2<double>(box.x + box.width, box.y);
        b_predicted.p3 = SCORE::Vec2<double>(box.x + box.width, box.y + box.height);
        b_predicted.p4 = SCORE::Vec2<double>(box.x, box.y + box.height);
        iou_score.push_back(computeIou(b, b_predicted));
        //Draw the mask for segmentation
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, img, predicted_mask, colors[i]);
    }

    //Written by Pietro Picardi
    //A hand was missed by the NN
    if(true_bb.size() > indices.size()) {
        for (int i = 0; i < true_bb.size(); i++) {
            Size labelSize = getTextSize("IoU: 0.000000", FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
            rectangle(img2, Point(0, 21 + i * (10 + labelSize.height)),
                      Point(labelSize.width, 2 * (21 + i * labelSize.height)),
                      Scalar(255, 255, 255), FILLED);
            putText(img2, "Missed hand", Point(0, 2 * (21 + i * labelSize.height) - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            //rectangle(img, Point(true_bb[i].p1.x, true_bb[i].p1.y), Point(true_bb[i].p3.x, true_bb[i].p3.y),Scalar(255, 255, 255), 1);
        }
    }

    //Written by Giacomo Gonella
    for (int i = indices.size() - 1; i >= 0; i--) {
        int idx = indices[i];
        Rect box = boxes[idx];
        //Draw the predicted bounding box
        rectangle(img2, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), colors[i], 2);
        //Display the label at the top of the bounding box
        Size labelSize = getTextSize("IoU: 0.000000", FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        rectangle(img2, Point(0, 21 + i * labelSize.height), Point(labelSize.width, 2 * (21 + i * labelSize.height)), colors[i], FILLED);
        putText(img2, "IoU: " + to_string(iou_score[i]), Point(0, 2 * (21 + i * labelSize.height) - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
    imshow("Detection image " + to_string(k), img2);
    waitKey(0);
    //imwrite("../results/Detection/det_" + to_string(k) + ".jpg", img2);
    //Compute the score for segmentation on the complete image
    pixel_accuracy = SCORE::computePixelAccuracy(predicted_mask, true_mask);
    //Display the score for segmentation on the complete image
    rectangle(img, Point(0, 0), Point(210, 20), Scalar(255, 255, 255), FILLED);
    putText(img, "Pixel accuracy: " + to_string(pixel_accuracy), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    imshow("Segmentation image " + to_string(k), img);
    waitKey(0);
    //imwrite("../results/Segmentation/seg_" + to_string(k) + ".jpg", img);
    destroyAllWindows();
} //postprocess

//Written by Pietro Picardi
void getTrueBoundingBoxes(string path, vector<SCORE::Box>* true_bb, Mat& img) {
    FILE* inDet;
    int x, y, width, height;
    inDet = fopen((path).c_str(), "r");
    if (inDet == NULL) {
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    while (!feof(inDet)) {
        SCORE::Box b;
        fscanf(inDet, "%d", &x);
        fscanf(inDet, "%d", &y);
        fscanf(inDet, "%d", &width);
        fscanf(inDet, "%d", &height);
        b.p1 = SCORE::Vec2<double>(x, y);
        b.p2 = SCORE::Vec2<double>(x + width, y);
        b.p3 = SCORE::Vec2<double>(x + width, y + height);
        b.p4 = SCORE::Vec2<double>(x, y + height);
        (*true_bb).push_back(b);
    }
    fclose(inDet);
}
