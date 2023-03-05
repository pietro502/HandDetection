//Entirely rewritten, edited and revised and adapted by Pietro Picardi.
//Revised by Paolo Bresolin and Giacomo Gonella

#ifndef SCORE_FILE_H
#define SCORE_FILE_H

#include <opencv2/highgui.hpp>
#include "math.h"
#include <assert.h>
#include <vector>

namespace SCORE {

    const double EPSILON = 1e-6;

    enum Wise
    {
        None,
        Clock,
        AntiClock
    };
    enum Location
    {
        Out,
        OnEdge,
        In
    };


    template <typename T>
    struct Vec2 {
        // Members
        union {
            struct
            {
                T x;
                T y;
            };
            T D[2];
        };

        // Constructors
        Vec2() : x(0), y(0) {}
        Vec2(T _x, T _y) : x(_x), y(_y) {}

        // Access component
        inline T& operator[](unsigned int i) {
            assert(i < 2);
            return D[i];
        }

        inline const T& operator[](unsigned int i) const {
            assert(i < 2);
            return D[i];
        }

        // Operations
        inline bool operator==(const Vec2 &p) const {
            return (abs(x - p.x) <= EPSILON && abs(y - p.y) <= EPSILON);
        }

        template<typename TT>
        inline Vec2 operator*(TT t) const {
            return Vec2(x * t, y * t);
        }

        template<typename TT>
        inline Vec2 operator/(TT t) const {
            return Vec2(x / t, y / t);
        }

        template<typename TT>
        inline Vec2& operator*=(TT t) {
            x *= t; y *= t;
            return *this;
        }

        template<typename TT>
        inline Vec2& operator/=(TT t) {
            x /= t; y /= t;
            return *this;
        }

        inline Vec2 operator+(const Vec2 &p) const {
            return Vec2(x + p.x, y + p.y);
        }

        inline Vec2 operator-(const Vec2 &p) const {
            return Vec2(x - p.x, y - p.y);
        }

        inline Vec2& operator+=(const Vec2 &p) {
            x += p.x; y += p.y;
            return *this;
        }

        inline Vec2& operator-=(const Vec2 &p) {
            x -= p.x; y -= p.y;
            return *this;
        }

        inline double dot(const Vec2 &p) const {
            return x * p.x + y * p.y;
        }

        inline double operator*(const Vec2 &p) const {
            return x * p.x + y * p.y;
        }

        inline double operator^(const Vec2 &p) const {
            return x * p.y - y * p.x;
        }

        inline double norm() const {
            return sqrt(x*x + y*y);
        }

        inline double normSquared() const {
            return x*x + y*y;
        }

        double distance(const Vec2 &p) const {
            return (*this - p).norm();
        }

        double squareDistance(const Vec2 &p) const {
            return (*this - p).normSquared();
        }

        double angle(const Vec2 &r) const {
            return acos( dot(r) / ( norm() * r.norm() ) );
        }

        double theta() const {
            return atan2(y, x);
        }
    };

    template <typename T>
    double computeNorm(const Vec2<T> &p) {
        return p.norm();
    }

    template <typename T>
    double computeNormSquared(const Vec2<T> &p) {
        return p.normSquared();
    }

    template <typename T, typename TT>
    inline Vec2<T> operator*(TT t, const Vec2<T>& v) {
        return Vec2<T>(v.x * t, v.y * t);
    }

    template <typename T>
    inline double computeDistance(const Vec2<T> &p1, const Vec2<T> &p2) {
        return p1.distance(p2);
    }

    template <typename T>
    inline double computeSquareDistance(const Vec2<T> &p1, const Vec2<T> &p2) {
        return p1.squareDistance(p2);
    }

    template <typename T>
    inline double computeAngle(const Vec2<T> &p1, const Vec2<T> &p2) {
        return p1.angle(p2);
    }

    template <typename T>
    inline double computeTheta(const Vec2<T> &p) {
        return p.theta();
    }

    typedef Vec2<double> Vec2d;
    typedef Vec2d Point;
    typedef std::vector<Point> Vertexes;

    class Segment {
    public:
        // Members
        Point p1;
        Point p2;

        // Constructors
        Segment() : p1(Point()), p2(Point()) {}
        Segment(const Point &_p1, const Point &_p2) : p1(_p1), p2(_p2) {}
        Segment(const Point _vert[2]) : p1(_vert[0]), p2(_vert[1]) {}

        // Methods
        bool isOnEdge(const Point &p) const;
        Point intersection(const Segment &segment, bool *bOnEdge = 0) const;
    };

    inline bool isOnEdge(const Segment &segment, const Point &p) {
        return segment.isOnEdge(p);
    }

    inline Point intersection(const Segment &segment1, const Segment &segment2, bool *bOnEdge = 0) {
        return segment1.intersection(segment2, bOnEdge);
    }


    class Box {
    public:
        // Members [in clockwise]
        Point p1;
        Point p2;
        Point p3;
        Point p4;

        // Constructors
        Box() : p1(Point()), p2(Point()), p3(Point()), p4(Point()) {}
        Box(const Point &_p1, const Point &_p2, const Point &_p3, const Point &_p4)
                : p1(_p1), p2(_p2), p3(_p3), p4(_p4) {}
        Box(const Point _vert[4])
                : p1(_vert[0]), p2(_vert[1]), p3(_vert[2]), p4(_vert[3]) {}

        // Methods
        void getVertList(Vertexes &_vert) const;
        double computeArea() const;
        Wise showWise() const;

    };


    // For any convex polygon
    double computeAreaEx(const Vertexes &C);
    Wise showWiseEx(const Vertexes &C);
    void beInSomeWiseEx(Vertexes &C, const Wise wiseType);
    Location computeLocationEx(const Vertexes &C, const Point &p);
    int interPtsEx(const Vertexes &C, const Segment &segment, Vertexes &pts);


    // For any convex polygon
    int findInterPointsEx(const Vertexes &C1, const Vertexes &C2, Vertexes &vert);
    int findInnerPointsEx(const Vertexes &C1, const Vertexes &C2, Vertexes &vert);


    // For convex quadrilateral
    int findInterPoints(const Box &B1, const Box &B2, Vertexes &vert);
    int findInnerPoints(const Box &B1, const Box &B2, Vertexes &vert);
    double computeAreaIntersection(const Box &B1, const Box &B2);
    double computeAreaUnion(const Box &B1, const Box &B2);
    double computeIou(const Box &B1, const Box &B2);

    //Pixel accuracy
    double computePixelAccuracy(cv::Mat&, cv::Mat&);
}
#endif // !_SCORE_H_FILE_
