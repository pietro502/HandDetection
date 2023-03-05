//Written and revised by Paolo Bresolin and Giacomo Gonella
//Revised by Pietro Picardi

#include "RegionGrowing.h"

using namespace std;
using namespace cv;

//Written by Giacomo Gonella
void clearNearPixels(Mat* matr, int x, int y, int range) {
    for (int i = x - range; i < x + range + 1; i++) {
        for (int j = y - range; j < y + range + 1; j++) {
            matr->at<uchar>(Point(i, j)) = 0;
        }
    }
}

//Written by Giacomo Gonella
vector<Point> selectInitialPoints(Mat mask) {
    Mat handPoints;
    vector<Point> points;
    threshold(mask, handPoints, 1, 255, THRESH_BINARY);
    /*
    imshow("mask", handPoints);
    waitKey(0);
    destroyAllWindows();
     */

    Point centralPoint = Point((int)handPoints.cols / 2, (int)handPoints.rows / 2);
    int colsRange = (int)handPoints.cols * 0.2;
    int rowsRange = (int)handPoints.rows * 0.2;

    Mat allowable = Mat::ones(handPoints.size(), handPoints.type());

    for (int i = centralPoint.x - colsRange; i < centralPoint.x + colsRange + 1; i++) {
        for (int j = centralPoint.y - rowsRange; j < centralPoint.y + rowsRange + 1; j++) {
            if (handPoints.at<uchar>(Point(i, j)) == 255 && allowable.at<uchar>(Point(i, j)) == 1) {
                //cout << "Point added: " << Point(i, j) << endl;
                points.push_back(Point(i, j));
                clearNearPixels(&allowable, i, j, 10);
            }
        }
    }
    if (points.size() == 0)
        points.push_back(centralPoint);
    return points;
}

//Written by Paolo Bresolin
/*
Create a mask containing the seed points stored in the vector of points $points.
Return the mask of seed points.
*/
Mat getSeedMaskFromPoints(const Mat& img, const vector<Point> points) {
    Mat mask = Mat::zeros(img.size(), CV_8U);
    for (Point point : points)
        mask.at<unsigned char>(point) = 255;
    return mask;
}//getSeedMaskFromPoints

//Written by Paolo Bresolin
/*
State if the point in $img at coordinates ($current_row, $current_col) is similar to the reference pixel $refference.
Return a bool which is true if and only if the two compared pixels are similar, according to threshold t.
*/
bool isSimilar(const Mat& img, const Vec3b& reference, int current_row, int current_col, int t) {
    bool c0 = (std::abs(img.at<Vec3b>(current_row, current_col)[0] - reference[0])) < t;
    bool c1 = (std::abs(img.at<Vec3b>(current_row, current_col)[1] - reference[1])) < t;
    bool c2 = (std::abs(img.at<Vec3b>(current_row, current_col)[2] - reference[2])) < t;
    return c0 && c1 && c2;
}//isSimilar

//Written by Paolo Bresolin
/*
Set neighborhood indices depending on the position of pixel ($current_row, $current_col) in $img.
Return:
    $indices[0] = starting row index of neighborhood;
    $indices[1] = last row index of neighborhood;
    $indices[2] = starting column index of neighborhood;
    $indices[3] = last column index of neighborhood;
*/
void neighborhoodIndices(const Mat& img, vector<int>& indices, int current_row, int current_col) {
    //Set starting row index of neighborhood
    indices[0] = current_row;
    if (indices[0] > 1)
        indices[0] -= 2;
    else if (indices[0] > 0)
        indices[0] -= 1;
    //Set last row index of neighborhood
    indices[1] = current_row;
    if (indices[1] < img.rows - 2)
        indices[1] += 2;
    else if (indices[1] < img.rows - 1)
        indices[1] += 1;
    //Set starting column index of neighborhood
    indices[2] = current_col;
    if (indices[2] > 1)
        indices[2] -= 2;
    else if (indices[2] > 0)
        indices[2] -= 1;
    //Set last column index of neighborhood
    indices[3] = current_col;
    if (indices[3] < img.cols - 2)
        indices[3] += 2;
    else if (indices[3] < img.cols - 1)
        indices[3] += 1;
}//neighborhoodIndices

//Written by Paolo Bresolin
/*
State if there are neighboring pixels to ($current_row, $current_col) with a seed in $mask.
Return a bool which is true if and only if there is at least a labelled neighbor in $mask.
*/
bool thereAreNeighbors(const Mat& img, const Mat& mask, int current_row, int current_col) {
    vector<int> neighbors(4);
    neighborhoodIndices(img, neighbors, current_row, current_col);
    for (int i = neighbors[0]; i <= neighbors[1]; i++)
        for (int j = neighbors[2]; j <= neighbors[3]; j++)
            if (mask.at<unsigned char>(i, j) == 255)
                return true;
    return false;
}//thereAreNeighbors

//Written by Paolo Bresolin
/*
Compute the reference value by considering the mean of the seed points in $seeds.
Return a Vec3b containing the mean of the values of the seeds in $seeds.
*/
Vec3b meanReference(const Mat& img, const vector<Point>& seeds) {
    Vec3b reference = img.at<Vec3b>(seeds[0]);
    for (int i = 1; i < seeds.size(); i++)
        for (int channel = 0; channel < 3; channel++)
            //Inefficient, but necessary to avoid overflow
            reference[channel] = (unsigned char) ((reference[channel] * i + img.at<Vec3b>(seeds[i])[channel]) / ((double) (i + 1)));
    return reference;
}//meanReference

//Written by Paolo Bresolin
/*
Perform region growing using as criterion of similarity the boolean values returned by functions $isSimilar and
$thereAreNeighbors. It exploits the vector of seed points $seeds, the threshold value $thresh and builds the mask
of labelled pixels $mask.
The structure of the function is simple to allow a better comprehension ad an easier debugging.
Return the built mask of labelled pixels $mask.
*/
void regionGrowing(const Mat& img, const vector<Point>& seeds, Mat& mask, const int thresh) {
    //Starting seed
    Point starting_seed(seeds[0]);
    //Pixel reference values
    Vec3b pixel_references = meanReference(img, seeds);
    //Coloring the starting seed
    mask.at<unsigned char>(starting_seed) = 255;
    //Region growing
    //From the first seed to the end of its row
    for (int j = starting_seed.x; j < img.cols; j++)
        if (isSimilar(img, pixel_references, starting_seed.y, j, thresh) && thereAreNeighbors(img, mask, starting_seed.y, j))
            mask.at<unsigned char>(starting_seed.y, j) = 255;
    //From the first seed to the beginning of its row
    for (int j = starting_seed.x - 1; j >= 0; j--)
        if (isSimilar(img, pixel_references, starting_seed.y, j, thresh) && thereAreNeighbors(img, mask, starting_seed.y, j))
            mask.at<unsigned char>(starting_seed.y, j) = 255;
    //From the row before the first seed to the first row
    for (int i = starting_seed.y - 1; i >= 0; i--)
        for (int j = 0; j < img.cols; j++)
            if (isSimilar(img, pixel_references, i, j, thresh) && thereAreNeighbors(img, mask, i, j))
                mask.at<unsigned char>(i, j) = 255;
    //From the row after the first seed to the last row
    for (int i = starting_seed.y + 1; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            if (isSimilar(img, pixel_references, i, j, thresh) && thereAreNeighbors(img, mask, i, j))
                mask.at<unsigned char>(i, j) = 255;
}//regionGrowing

//Written by Paolo Bresolin
/*
Given a sub - image $b_box, a mask of labelled pixels $maskand a color, colors the pixels of& b_box labelled in $mask
with color $color.
Return $b_box with colored pixels according to $mask.
*/
void colorBoundingBox(Mat& b_box, const Mat& mask, const Scalar color) {
    for (int i = 0; i < b_box.rows; i++)
        for (int j = 0; j < b_box.cols; j++)
            if (mask.at<unsigned char>(i, j) == 255)
                for (int channel = 0; channel < 3; channel++)
                    b_box.at<Vec3b>(i, j)[channel] = color[channel];
}//colorBoundingBox
