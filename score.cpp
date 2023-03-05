//Entirely rewritten, edited and revised and adapted by Pietro Picardi.
//Revised by Paolo Bresolin and Giacomo Gonella


#include "score.h"
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace SCORE {

    bool Segment::isOnEdge(const Point &p) const {
        if (p1 == p2)
            return (p == (p1 + p2) / 2.0);

        Point pp1 = p - p1;
        Point pp2 = p - p2;

        if ( abs(pp1^pp2) < EPSILON &&
            pp1*pp2 < EPSILON )
            return true;
        else
            return false;
    }

    Point Segment::intersection(const Segment &segment, bool *bOnEdge) const {
        Point pInter(0,0);
        bool bOn = false;

        if (p1 == p2 && segment.p1 == segment.p2){
            // Both lines are actually points.
            bOn =((p1 + p2) / 2.0 == (segment.p1 + segment.p2) / 2.0);
            if (bOn)
                pInter = (p1 + p2 + segment.p1 + segment.p2) / 4.0;
        }
        else if (p1 == p2) {
            // This line is actually a point.
            bOn = segment.isOnEdge((p1 + p2) / 2.0);
            if (bOn)
                pInter = (p1 + p2) / 2.0;
        }
        else if (segment.p1 == segment.p2) {
            // The input line is actually a point.
            bOn = isOnEdge((segment.p1 + segment.p2) / 2.0);
            if (bOn)
                pInter = (segment.p1 + segment.p2) / 2.0;
        }
        else {
            // Normal cases.
            Point a12 = p2 - p1;
            Point b12 = segment.p2 - segment.p1;
            double ang = computeAngle(a12, b12);
            if (ang < EPSILON || abs(3.141592653 - ang) < EPSILON)
                bOn = false; // Collinear!!
            else {
                // a1_x + m*a12_x = b1_x + n*b12_x
                // a1_y + m*a12_y = b1_y + n*b12_y
                // n = ( (a1_y-b1_y)*a12_x - (a1_x-b1_x)*a12_y ) / (a12_x*b12_y - b12_x*a12_y)
                // m = ( (a1_y-b1_y)*b12_x - (a1_x-b1_x)*b12_y ) / (a12_x*b12_y - b12_x*a12_y)
                // 0 < m < 1
                // 0 < n < 1
                double abx = p1.x - segment.p1.x;
                double aby = p1.y - segment.p1.y;
                double ab = a12.x*b12.y - b12.x*a12.y;
                assert( abs(ab) > EPSILON );
                double n = (aby*a12.x - abx*a12.y) / ab;
                double m = (aby*b12.x - abx*b12.y) / ab;

                if (n >= -EPSILON && n - 1.0 <= EPSILON &&
                    m >= -EPSILON && m - 1.0 <= EPSILON) {
                    Point ip1 = p1 + m*a12;
                    Point ip2 = segment.p1 + n*b12;
                    pInter = (ip1 + ip2) / 2.0;
                    bOn = true;
                }
                else
                    bOn = false;
            }
        }
        if (bOnEdge != 0)
            *bOnEdge = bOn;
        return pInter;
    }

    void Box::getVertList(Vertexes &_vert) const {
        Vertexes vertTemp;
        vertTemp.reserve(4);
        vertTemp.push_back(p1);
        vertTemp.push_back(p2);
        vertTemp.push_back(p3);
        vertTemp.push_back(p4);
        _vert.swap(vertTemp);
    }

    double Box::computeArea() const {
        Vertexes vertTemp;
        getVertList(vertTemp);

        return computeAreaEx(vertTemp);
    }

    Wise Box::showWise() const {
        Vertexes vertTemp;
        getVertList(vertTemp);
        return showWiseEx(vertTemp);
    }

    double computeAreaEx(const Vertexes &C) {
        if ( showWiseEx(C) == None )
            return -1.0;

        double sArea = 0.0;
        const int N = C.size();
        if ( N > 2 ) {
            const Point &p0 = C.at(0);
            for (int i = 1; i < N-1; ++i) {
                const Point &p1 = C.at(i);
                const Point &p2 = C.at(i + 1);
                Point p01 = p1 - p0;
                Point p02 = p2 - p0;
                sArea += abs(p01^p02)*0.5;
            }
        }
        return sArea;
    }

    Wise showWiseEx(const Vertexes &C) {
        Wise wiseType = None;
        const int N = C.size();

        if ( N > 2 ) {
            Point p0 = C.at(N - 1);
            Point p1 = C.at(0);
            Point p2 = C.at(1);
            Point p01 = p1 - p0;
            Point p12 = p2 - p1;
            if ( ( abs(p01^p12) <= EPSILON ) && p01*p12 < 0.0 )
                return None;
            else
                wiseType = ( p01^p12 ) > 0.0 ? AntiClock : Clock;

            const double flip = (wiseType == Clock) ? 1.0 : -1.0;
            for (int i = 1; i < N ; ++i) {
                p0 = C.at((i-1)%N);
                p1 = C.at(i%N);
                p2 = C.at((i+1)%N);
                p01 = p1 - p0;
                p12 = p2 - p1;
                if ( ( p01^p12 ) * flip > 0.0 ||
                    ( ( abs(p01^p12) <= EPSILON ) && p01*p12 < 0.0 ) )
                    return None;
            }
        }
        return wiseType;
    }

    typedef std::pair<double, Point> AngPoint;

    bool angIncrease(const AngPoint &p1, const AngPoint &p2) {
        return p1.first < p2.first;
    }

    bool angDecrease(const AngPoint &p1, const AngPoint &p2) {
        return p1.first > p2.first;
    }

    void beInSomeWiseEx(Vertexes &C, const Wise wiseType) {
        if ( wiseType != None ) {
            const int N = C.size();
            if ( N > 2 ) {
                Point pO(0.0,0.0);
                for (int i = 0; i < N; ++i)
                    pO += C.at(i);
                pO /= N;
                std::vector<AngPoint> APList;
                APList.reserve(N);
                for (int i = 0; i < N; ++i) {
                    Point op = C.at(i) - pO;
                    double ang = op.theta();
                    APList.push_back(AngPoint(ang, C.at(i)));
                }
                if ( wiseType == AntiClock )
                    std::sort(APList.begin(), APList.end(), angIncrease);
                else
                    std::sort(APList.begin(), APList.end(), angDecrease);
                Vertexes vertTemp;
                for (int i = 0; i < N; ++i)
                    vertTemp.push_back(APList.at(i).second);
                C.swap(vertTemp);
            }
        }
    }

    Location locationEx(const Vertexes &C, const Point &p) {
        const int N = C.size();
        // Special cases.
        if (N == 0)
            return Out;
        if (N == 1) {
            if (C[0] == p)
                return In;
            else
                return Out;
        }
        if (N == 2) {
            if ( isOnEdge(Segment(C[0],C[1]), p ) )
                return OnEdge;
            else
                return Out;
        }

        // Normal cases.
        // Check OnEdge.
        for (int i = 0; i < N; ++i) {
            if ( isOnEdge(Segment(C[i % N],C[ ( i + 1 ) % N]), p ) )
                return OnEdge;
        }
        // Check Outside.
        Point pO(0.0,0.0);
        for (int i = 0; i < N; ++i) {
            pO += C[i];
        }
        pO /= N;
        Segment op(pO,p);
        bool bIntersection = true;
        for (int i = 0; i < N; ++i) {
            intersection(Segment(C[i%N],C[(i+1)%N]),op,&bIntersection);
            if (bIntersection)
                return Out;
        }
        return In;
    }
    int interPtsEx(const Vertexes &C, const Segment &line, Vertexes &pts) {
        Vertexes vertTemp;
        const int N = C.size();
        bool bIntersection = false;
        for (int i = 0; i < N; ++i) {
            Point p = intersection(Segment(C[i%N],C[(i+1)%N]),line,&bIntersection);
            if (bIntersection)
                vertTemp.push_back(p);
        }
        pts.swap(vertTemp);
        return pts.size();
    }

    int findInterPointsEx(const Vertexes &C1, const Vertexes &C2, Vertexes &vert) {
        Vertexes _vert;
        const int N = C2.size();
        for (int i = 0; i < N; ++i) {
            Vertexes pts;
            interPtsEx(C1,Segment(C2[i % N],C2[ (i + 1) % N]), pts );
            for (int i = 0; i < pts.size(); ++i)
                _vert.push_back(pts.at(i));
        }
        vert.swap(_vert);
        return vert.size();
    }

    int findInnerPointsEx(const Vertexes &C1, const Vertexes &C2, Vertexes &vert) {
        Vertexes _vert;
        for (int i = 0; i < C2.size(); ++i) {
            if (locationEx(C1,C2[i]) != Out)
                _vert.push_back(C2[i]);
        }
        vert.swap(_vert);
        return vert.size();
    }

    int findInterPoints(const Box &B1, const Box &B2, Vertexes &vert) {
        Vertexes V1, V2;
        B1.getVertList(V1);
        B2.getVertList(V2);
        return findInterPointsEx(V1,V2,vert);
    }

    int findInnerPoints(const Box &B1, const Box &B2, Vertexes &vert) {
        Vertexes V1, V2;
        B1.getVertList(V1);
        B2.getVertList(V2);
        return findInnerPointsEx(V1,V2,vert);
    }

    double computeAreaIntersection(const Box &B1, const Box &B2) {
        if (B1.showWise() == None ||
            B2.showWise() == None )
            return -1.0;

        Vertexes interVert;
        Vertexes innerVert12;
        Vertexes innerVert21;
        Vertexes allVerts;
        //---------------
        findInterPoints(B1, B2, interVert);
        findInnerPoints(B1, B2, innerVert12);
        findInnerPoints(B2, B1, innerVert21);
        //---------------
        // TODO : Check conditions
        for (int i = 0; i < interVert.size(); ++i)
            allVerts.push_back(interVert.at(i));
        for (int i = 0; i < innerVert12.size(); ++i)
            allVerts.push_back(innerVert12.at(i));
        for (int i = 0; i < innerVert21.size(); ++i)
            allVerts.push_back(innerVert21.at(i));

        if (allVerts.empty())
            return 0.0;
        else {
            assert(allVerts.size() >= 3);
            beInSomeWiseEx(allVerts, Clock);
            if ( showWiseEx(allVerts) == None )
                return -1.0;
            else
                return computeAreaEx(allVerts);
        }
    }

    double computeAreaUnion(const Box &B1, const Box &B2) {
        return B1.computeArea()+B2.computeArea()-computeAreaIntersection(B1,B2);
    }

    double computeIou(const Box &B1, const Box &B2) {
        return computeAreaIntersection(B1,B2)/computeAreaUnion(B1,B2);
    }

    double computePixelAccuracy(Mat& predicted_mask, Mat& true_mask) {
        double counter = 0;
        cvtColor(true_mask, true_mask, COLOR_BGR2GRAY);
        cvtColor(predicted_mask, predicted_mask, COLOR_BGR2GRAY);

        for (int i = 0; i < true_mask.rows; i++)
            for (int j = 0; j < true_mask.cols; j++)
                if (true_mask.at<uchar>(i, j) == predicted_mask.at<uchar>(i, j))
                    counter += 1;

        return counter / (true_mask.cols * true_mask.rows);
    }

}
