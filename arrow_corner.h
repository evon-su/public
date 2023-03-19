#include <vector>
#include "mat.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>


typedef Eigen::Vector2f Coef;
typedef Eigen::Vector2f Point;


enum class ArrowType : unsigned char
{
    Arrow_SIMPLE,   // Standard Arrow Of Linear
    Arrow_LEFT,     // Turn Left Arrow
    Arrow_RIGHT,    // Turn Right Arrow
    Others
};

struct ArrowInfo
{
    std::vector<Eigen::Vector2f> corners = {{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f}};     // length of 7

    ArrowType arrowType;
    float score;
};

struct ArrowLine
{
    Coef coef;
    Point end1;
    Point end2;
    int ptsNumbers = -1;
    float meanErr = -1.f;
    float theta = -1.f;
};


class ArrowCorner
{
private:
    // Algorithm
    ArrowInfo findCorners(std::vector<Eigen::Vector2f>& arrow);
    std::vector<ArrowInfo> findCorners(std::vector<std::vector<Eigen::Vector2f>>& arrows);
    std::vector<ArrowLine> linesByRansac(const std::vector<Eigen::Vector2f>& points, int iterations, int randNumbers, float sigma, int fitOrder);
    std::vector<ArrowLine> ptLinesByRansac(const std::vector<Eigen::Vector2f>& points, int fixPt, int iterations, int randNumbers, float sigma, int fitOrder, float arrowLen);
    ArrowLine fitLineByRansac(std::vector<Eigen::Vector2f>::const_iterator start, std::vector<Eigen::Vector2f>::const_iterator end, int iterations, int randNumbers, float sigma, int fitOrder);
    std::vector<int> ransacToTerminal(const std::vector<Eigen::Vector2f>& points);
    std::vector<int> terminalToSides(const std::vector<Eigen::Vector2f>& points, int terminal1, int terminal2);
    ArrowType findArrowType(std::vector<Eigen::Vector2f>& points, int T1, int T2, int S1, int S2);
    std::vector<Eigen::Vector2f> terminalToCorners(int terminal1, int terminal2, const std::vector<Eigen::Vector2f>& points);

    // Preprocessing
    std::vector<std::vector<Eigen::Vector2f>> arrowsTo3D(const std::vector<std::vector<Eigen::Vector2f>>& arrows2D);
    std::vector<Eigen::Vector2f> pointsToAvg(const std::vector<Eigen::Vector2f>& points, float k1);

    // Transform
    ArrowInfo linesToInfo(const std::vector<ArrowLine>& lines, int lineNum);
    
    // Util
    float ptToLineDistance(const Eigen::Vector2f& pt, const Coef& lineCoef);
    Eigen::Vector2f linesToInterscep(const Coef& line1, const Coef& line2);
    float pointsToDist(const Eigen::Vector2f pt1, const Eigen::Vector2f pt2);
    float areaByPolygon(const std::vector<Eigen::Vector2f>& polygon);
    Point pointsToCenter(Eigen::Vector2f pt1, Eigen::Vector2f pt2);
    float coefToTheta(Coef coef1, Coef coef2);
    Coef periodsToCoef(int pos1, int pos2, const std::vector<Eigen::Vector2f>& points);
    std::vector<Eigen::Vector2f> periodsToPoints(int pos1, int pos2, const std::vector<Eigen::Vector2f>& points);
    Coef ptSlopToCoef(Eigen::Vector2f pt, float slop);
    Coef pt2ToCoef(Eigen::Vector2f pt1, Eigen::Vector2f pt2);
    std::vector<cv::Point> inversePts(const std::vector<cv::Point>& pts);
    Eigen::Vector2f pt3(Eigen::Vector2f& inputPt);
    std::vector<Eigen::Vector2f> arrowTo3D_2(const std::vector<cv::Point>& arrow2D, int threshold);
    std::vector<std::vector<Eigen::Vector2f>> arrowsTo3D_2(const std::vector<std::vector<Eigen::Vector2f>>& arrows2D, int threshold);
    std::vector<cv::Point2f> arrowTo2D_cv(const std::vector<Eigen::Vector2f>& inputPts3d);
    Eigen::Vector4f polyfit(const std::vector<std::vector<Eigen::Vector2f>>& lanesGroup, int order);
    cv::Point2f pt3_inv_cv(const cv::Point2f& inputPt);
    std::vector<cv::Point2f> eigen2cvPoint(std::vector<Eigen::Vector2f>& input);

    // Members
    float m_sigma = 0.08;           // It Should Depend On Arrow Size 
    float m_movingAvg_k = 5;
    float m_meanArrowWidth_line = 0.938;
    float m_arrow_orth_theta_thres = 1.2;


public:
    // Constructor
    ArrowCorner() = default; 
    ArrowCorner(const ArrowCorner& other);
    ~ArrowCorner() {};
    ArrowCorner& operator=(const ArrowCorner& other);

    // API
    std::vector<ArrowInfo> process(std::vector<std::vector<Eigen::Vector2f>>& arrows);     // arrowNum=3 for gwtv
    std::vector<cv::Point2f> process(const std::vector<cv::Point>& arrow);


};  // End of ArrowCorner

