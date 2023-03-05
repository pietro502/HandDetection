//Written and revised by Paolo Bresolin and Giacomo Gonella.
//Revised by Pietro Picardi

#ifndef REGION_GROWING_FILE_H
#define REGION_GROWING_FILE_H

#include <iostream>
#include <opencv2/imgproc.hpp>

//Written by Giacomo Gonella
void clearNearPixels(cv::Mat*, int, int, int);

//Written by Giacomo Gonella
std::vector<cv::Point> selectInitialPoints(cv::Mat);

//Written by Paolo Bresolin
cv::Mat getSeedMaskFromPoints(const cv::Mat&, const std::vector<cv::Point>);

//Written by Paolo Bresolin
bool isSimilar(const cv::Mat&, const cv::Vec3b&, int, int, int);

//Written by Paolo Bresolin
void neighborhoodIndices(const cv::Mat&, std::vector<int>&, int, int);

//Written by Paolo Bresolin
bool thereAreNeighbors(const cv::Mat&, const cv::Mat&, int, int);

//Written by Paolo Bresolin
cv::Vec3b meanReference(const cv::Mat&, const std::vector<cv::Point>&);

//Written by Paolo Bresolin
void regionGrowing(const cv::Mat&, const std::vector<cv::Point>&, cv::Mat&, const int);

//Written by Paolo Bresolin
void colorBoundingBox(cv::Mat&, const cv::Mat&, const cv::Scalar);

#endif
