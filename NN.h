//Written and revised by Paolo Bresolin, Giacomo Gonella and Pietro Picardi

#ifndef NN_FILE_H
#define NN_FILE_H

//Written by Paolo Bresolin
std::vector<std::string> getOutputsNames(const cv::dnn::Net&);

//Written by Giacomo Gonella and Pietro Picardi
void drawPred(int, float, int, int, int, int, cv::Mat&, cv::Mat&, const cv::Scalar);

//Written by Paolo Bresolin, Giacomo Gonella and Pietro Picardi
void postprocess(cv::Mat&, const std::vector<cv::Mat>&, const std::vector<SCORE::Box>, cv::Mat&, const int k);

//Written by Pietro Picardi
void getTrueBoundingBoxes(std::string, std::vector<SCORE::Box>*, cv::Mat&);

#endif
