#include "arrow_corner.h"
#include <unordered_set>
#include <cmath>
// #include <fstream>
#include "util.h"

using namespace std;

#define showdata 1

float PI = 3.1415926f;


ArrowCorner::ArrowCorner(const ArrowCorner& other) {
    (*this) = other;
}

std::vector<cv::Point2f> ArrowCorner::process(const std::vector<cv::Point>& arrow)
{
    std::vector<cv::Point2f> outArrow;
    if (arrow.size() == 0) {
        return outArrow;
    }
    
    // Inverse
    std::vector<cv::Point> arrow_inv = inversePts(arrow);

    // Project to 3D
    auto arrows3D = arrowTo3D_2(arrow_inv, 50);

    // Find Corners
    auto infos = findCorners(arrows3D);
    std::vector<Eigen::Vector2f> polygon = infos.corners;

    // Projec to 2D
    if (polygon[0].x() > -100.f) outArrow = arrowTo2D_cv(polygon);
    else outArrow = eigen2cvPoint(polygon);

    return outArrow;
}

std::vector<ArrowInfo> ArrowCorner::process(std::vector<std::vector<Eigen::Vector2f>>& arrows)
{
    std::vector<ArrowInfo> infos;
    std::sort(arrows.begin(), arrows.end(), [](const std::vector<Eigen::Vector2f>& a, 
                                                 const std::vector<Eigen::Vector2f>& b) {
            return a.size() > b.size(); });
    if (arrows.size() == 0 || arrows[0].size() < 100) {
        return infos;
    }

    // Project to 3D
    // auto arrows3D = arrowsTo3D(arrows);
    auto arrows3D = arrowsTo3D_2(arrows, 100);

    // Find Corners
    infos = findCorners(arrows3D);

    return infos;
}

ArrowInfo ArrowCorner::findCorners(std::vector<Eigen::Vector2f>& arrow)
{
    ArrowInfo info;
    auto arrowAvg = arrow;
    auto terminals = ransacToTerminal(arrowAvg);
    auto sides = terminalToSides(arrowAvg, terminals[0], terminals[1]);

    // Check Arrow Type ( Line Or Curve )
    // --> To Be Done
    ArrowType type = findArrowType(arrowAvg, terminals[0], terminals[1], sides[0], sides[1]);

    // Get Corners
    if (type == ArrowType::Arrow_SIMPLE) {
        auto Corners = terminalToCorners(terminals[0], terminals[1], arrowAvg);
        info.corners = Corners;
        info.arrowType = ArrowType::Arrow_SIMPLE;
    }

    return info;
}

std::vector<ArrowInfo> ArrowCorner::findCorners(std::vector<std::vector<Eigen::Vector2f>>& arrows)
{
    std::vector<ArrowInfo> infos;

    for (auto& arrow: arrows)
    {
        auto info = findCorners(arrow);
        infos.push_back(info);
    }

    return infos;
}

std::vector<int> ArrowCorner::ransacToTerminal(const std::vector<Eigen::Vector2f>& points)
{
    auto randN = [] (int max)->Eigen::Vector2i {
        if (max > 1)
        {
            int a = rand() % max;
            int b = rand() % max;
            while (a == b) {
                b = rand() % max;
            }
            return {a, b};
        }
        else
            return {0, 0};
    };

    int iterations = 500;
    Eigen::Vector2i terminalPts{0, 0};
    float maxLen = 0;

    for (int i=0; i< iterations; ++i)
    {
        Eigen::Vector2i randPairs = randN(points.size() - 1);

        float ptsLen = pointsToDist(points[randPairs[0]], points[randPairs[1]]);
        if (ptsLen > maxLen) {
            maxLen = ptsLen;
            terminalPts = randPairs;
        }

    }
    return {terminalPts[0], terminalPts[1]};
}

std::vector<int> ArrowCorner::terminalToSides(const std::vector<Eigen::Vector2f>& points, int terminal1, int terminal2)
{
    int term1 = std::min(terminal1, terminal2);
    int term2 = std::max(terminal1, terminal2);

    Coef coef0 = pt2ToCoef(points[term1], points[term2]);
    float maxDis1=0, maxDis2=0;
    int maxPt1=term1, maxPt2=term2;

    // Find First Corner
    for (int i=term1+1; i<term2; ++i) {
        float dis = ptToLineDistance(points[i], {coef0[0], coef0[1]});
        if (fabs(dis) > fabs(maxDis1)) {
            maxDis1 = dis;
            maxPt1 = i;
        }
    }

    // Find Second Corner
    for (int i=term2+1; i<points.size(); ++i) {
        float dis = ptToLineDistance(points[i], {coef0[0], coef0[1]});
        if (fabs(dis) > fabs(maxDis2)) {
            maxDis2 = dis;
            maxPt2 = i;
        }
    } for (int i=0; i<term1; ++i) {
        float dis = ptToLineDistance(points[i], {coef0[0], coef0[1]});
        if (fabs(dis) > fabs(maxDis2)) {
            maxDis2 = dis;
            maxPt2 = i;
        }
    }

    return {maxPt1, maxPt2};
}

ArrowType ArrowCorner::findArrowType(std::vector<Eigen::Vector2f>& points, int T1, int T2, int S1, int S2)
{
    if (T1 == T2 || S1 == S2)
        return ArrowType::Others;

    int term1 = std::min(T1, T2);
    int term2 = std::max(T1, T2);
    // Variables
    bool near_paraller=false, side_valid=true;
    float dist_TT=0, dist_SS=0;

    // Check By Compare Linear Direction & TT Direction
    Eigen::Vector4f coef0_ = polyfit({points}, 1);
    Coef coef0 = {coef0_[0], coef0_[1]};
    Coef coefTT = pt2ToCoef(points[T1], points[T2]);
    float theta = coefToTheta(coef0, coefTT);
    if (fabs(theta) < PI / 8.f) {  ////////////////////// Identify Near Parallel
        near_paraller = true;
    }

    // Check By Length
    dist_SS = pointsToDist(points[S1], points[S2]);
    dist_TT = pointsToDist(points[T1], points[T2]);

    // Check By Angle
    Coef coefSS = pt2ToCoef(points[S1], points[S2]);
    float arrowTheta = coefToTheta(coefSS, coef0);

    // Check By Points Trend
    float sign1 = ptToLineDistance(points[term1], coef0);
    for (int i=term1+1; i<term2; ++i) {
        if (sign1 * ptToLineDistance(points[i], coef0) < 0) {
            side_valid = false;
            break;
        }
    } if (side_valid) {
        for (int i=term2; i<points.size(); ++i) {
            if (sign1 * ptToLineDistance(points[i], coef0) < 0) {
                side_valid = false;
                break;
            }
        }
    } if (side_valid) {
        for (int i=0; i<term1; ++i) {
            if (sign1 * ptToLineDistance(points[i], coef0) < 0) {
                side_valid = false;
                break;
            }
        }
    }

    // Determina Arrow Type
    if (near_paraller && 
        side_valid && 
        // fabs(dist_SS - m_meanArrowWidth_line) < m_meanArrowWidth_line/5.f && 
        arrowTheta > m_arrow_orth_theta_thres) {
        return ArrowType::Arrow_SIMPLE;

    } else {
        // cout<<"============= type ============="<<endl;
        // cout<<near_paraller<<", "<<side_valid<<", "<<fabs(dist_SS - m_meanArrowWidth_line)<<", " << m_meanArrowWidth_line/5.f<<", "<<arrowTheta <<", "<< m_arrow_orth_theta_thres<<endl;

        return ArrowType::Others;
    }
}

std::vector<Eigen::Vector2f> ArrowCorner::terminalToCorners(int terminal1, int terminal2, const std::vector<Eigen::Vector2f>& points)
{
    std::vector<Eigen::Vector2f>::const_iterator it = points.cbegin();

    auto isNear = [&] (const Eigen::Vector2f& pt, Eigen::Vector2f& coef, float threshold)->bool {
        float dis = ptToLineDistance(pt, coef);
        if (fabs(dis) < fabs(threshold)) {
            return true;
        } else {
            return false;
        }
    };
    auto periodToErr = [&] (int pos1, int pos2)->float {
        if (pos1 == pos2) {throw std::runtime_error("pos1 cannot equal to pos2 !"); }
        
        Coef coef = periodsToCoef(pos1, pos2, points);
        
        float meanErr = 0;
        if (pos2 > pos1) {
            for (int i=pos1; i<pos2; ++i) {
                float err = ptToLineDistance(*(it+i), coef);
                meanErr += fabs(err);
            }
            return meanErr / (pos2 - pos1);
        } else {
            for (int i=pos1; i<points.size(); ++i) {
                float err = ptToLineDistance(*(it+i), coef);
                meanErr += fabs(err);
            } for (int i=0; i<pos2; ++i) {
                float err = ptToLineDistance(*(it+i), coef);
                meanErr += fabs(err);
            }
            return meanErr / (points.size() - pos1 -1 + pos2);
        }
    };
    auto refineP0 = [&](Eigen::Vector2i period1, Eigen::Vector2i period2, float threshold)->Point {
        auto pts1 = periodsToPoints(period1[0], period1[1], points);
        auto pts2 = periodsToPoints(period2[0], period2[1], points);
        
        ArrowLine line1 = fitLineByRansac(pts1.begin(), pts1.end(), 100, 2, threshold, 1);
        ArrowLine line2 = fitLineByRansac(pts2.begin(), pts2.end(), 100, 2, threshold, 1);

        return linesToInterscep(line1.coef, line2.coef);
    };
    auto replaceP =[](Point& newPt, std::vector<Eigen::Vector2f>::iterator out){
        if (std::isfinite(newPt.x()) && std::isfinite(newPt.y())) {
            *out = {newPt.x(), newPt.y()};
        }
    };
    auto refine34 = [&](Coef& lineCoef, Coef& coefOrthog, std::vector<Eigen::Vector2f>::iterator out){
        auto pt = linesToInterscep(lineCoef, coefOrthog);
        replaceP(pt, out);
    };
    auto refine2345 = [&](int start, int end, float threshold, Coef& coef0, Coef& coef16, std::vector<Eigen::Vector2f>::iterator out1, 
    Coef& coef_orthog, std::vector<Eigen::Vector2f>::iterator out2)
    {
        auto pts = periodsToPoints(start, end, points);
        ArrowLine line = fitLineByRansac(pts.begin(), pts.end(), 100, 2, m_sigma/2.f, 1);

        if (fabs(coefToTheta(coef0, line.coef)) < PI/8.f) /*make sure the line is correct*/
        {
            auto pt = linesToInterscep(coef16, line.coef);
            replaceP(pt, out1);
        }
        refine34(line.coef, coef_orthog, out2);
    };

    int term1 = std::min(terminal1, terminal2);
    int term2 = std::max(terminal1, terminal2);

    Coef coef0 = pt2ToCoef(points[term1], points[term2]);
    float maxDis1=0, maxDis2=0;
    int maxPt1=0, maxPt2=0;

    // Find First Corner
    for (int i=term1+1; i<term2; ++i) {
        float dis = ptToLineDistance(points[i], coef0);
        if (fabs(dis) > fabs(maxDis1)) {
            maxDis1 = dis;
            maxPt1 = i;
        }
    }

    // Find Second Corner
    for (int i=term2+1; i<points.size(); ++i) {
        float dis = ptToLineDistance(points[i], coef0);
        if (fabs(dis) > fabs(maxDis2)) {
            maxDis2 = dis;
            maxPt2 = i;
        }
    } for (int i=0; i<term1; ++i) {
        float dis = ptToLineDistance(points[i], coef0);
        if (fabs(dis) > fabs(maxDis2)) {
            maxDis2 = dis;
            maxPt2 = i;
        }
    }

    // Filter By Confidence
    

    // Determing Point "0" TEST
    float err1 = periodToErr(term1, maxPt1);
    float err2 = periodToErr(maxPt1, term2);
    float err3 = periodToErr(term2, maxPt2);
    float err4 = periodToErr(maxPt2, term1);

    // OUTPUT
    int arrowType = -1;    // 0: Cusp is Term1, 1: Cusp is Term2
    int P0=-1, P1=-1, P6=-1;
    std::vector<Eigen::Vector2f> output={{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f},{-100.f,-100.f}};
    if (err1 < err2 && err1 < err3 && err4 < err2 && err4 < err3 && err1 < 0.03 && err4 < 0.03) {

        // Cusp is term1
        P0 = term1, P1 = maxPt1, P6 = maxPt2;

        // Refine P0
        auto pt_refine = refineP0({term1, maxPt1}, {maxPt2, term1}, m_sigma/2.f);
        if (fabs(pointsToDist(points[term1], pt_refine)) < 0.4)  // Refine Only When It's Small Difference. If It Differ Large, WON'T Refine It.
        {
            output[0] = pt_refine;
        } else {
            output[0] = points[term1];
        }

        // Refine P1 & P6
        auto pts1 = periodsToPoints(P0, P1, points);
        auto pts2 = periodsToPoints(P6, P0, points);
        ArrowLine line01 = fitLineByRansac(pts1.cbegin(), pts1.cend(), 100, 2, m_sigma/2.f, 1);
        ArrowLine line06 = fitLineByRansac(pts2.cbegin(), pts2.cend(), 100, 2, m_sigma/2.f, 1);
        // P1
        for (int i=maxPt1; i<term2; ++i) {
            if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                P1 = i;
            }
        }
        // P6
        if (maxPt2 > term2) {
            for (int i=maxPt2; i>term2; --i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P6 = i;
                }
            }
        } else {
            for (int i=maxPt2; i>0; --i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P6 = i;
                }
            } for (int i=points.size()-1; i>term2; --i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P6 = i;
                }
            }
        }

        output[1] = points[P1];
        output[6] = points[P6];

        arrowType = 1;
    } else if (err2 < err1 && err2 < err4 && err3 < err1 && err3 < err4 && err2 < 0.03 && err3 < 0.03) {

        P0 = term2, P1 = maxPt2, P6 = maxPt1;

        // Refine P0
        auto pt_refine = refineP0({maxPt1, term2}, {term2, maxPt2}, m_sigma/2.f);
        if (fabs(pointsToDist(points[term2], pt_refine)) < 0.4)  // Refine Only When It's Small Difference. If It Differ Large, WON'T Refine It.
        {
            output[0] = pt_refine;
        } else {
            output[0] = points[term2];
        }

        // Refine P1 & P6
        auto pts1 = periodsToPoints(P0, P1, points);
        auto pts2 = periodsToPoints(P6, P0, points);
        ArrowLine line01 = fitLineByRansac(pts1.cbegin(), pts1.cend(), 100, 2, m_sigma/2.f, 1);
        ArrowLine line06 = fitLineByRansac(pts2.cbegin(), pts2.cend(), 100, 2, m_sigma/2.f, 1);
        // P1
        if (term1 > maxPt2) {
            for (int i=maxPt2; i<term1; ++i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P1 = i;
                }
            }
        } else {
            for (int i=maxPt2; i<points.size(); ++i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P1 = i;
                }
            } for (int i=0; i<term1; ++i) {
                if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                    P1 = i;
                }
            }
        }
        // P6
        for (int i=maxPt1; i< term1; ++i) {
            if (fabs(ptToLineDistance(points[i], line01.coef)) < m_sigma / 2.f) {
                P6 = i;
            }
        }

        output[1] = points[P1];
        output[6] = points[P6];

        arrowType = 2;
    } else {
        output[0] = {-200.f, -200.f};   // It Means NOT NORMAL
        return output;
    }
    
    // Find Rectangle Cornerss
    Coef coef;

    coef = pt2ToCoef(points[maxPt1], points[maxPt2]);

    int P2, P3, P4, P5;
    if (arrowType == 1) {
        P2=P3=P4=P5=term2;
    } else if (arrowType == 2) {
        P2=P3=P4=P5=term1;
    }

    float threshold = m_sigma*2.f;
    if (arrowType == 1) {
        for (int i=maxPt1; i<term2; ++i) {
            float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
            if (fabs(dis) < threshold) {
                P2 = i;
            }
        }
        if (maxPt2 > term2) {
            for (int i=maxPt2; i>term2; --i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P5 = i;
                }
            }
        } else {
            for (int i=maxPt2; i>0; --i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P5 = i;
                }
            } for (int i=points.size(); i>term2; --i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P5 = i;
                }
            }
        }

    } else if (arrowType == 2) {
        for (int i=maxPt1; i>term1; --i) {
            float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
            if (fabs(dis) < threshold) {
                P5 = i;
            }
        }
        if (term1 > maxPt2) {
            for (int i=maxPt2; i<term1; ++i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P2 = i;
                }
            }
        } else { 
            for (int i=maxPt2; i<points.size(); ++i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P2 = i;
                }
            } for (int i=0; i<term1; ++i) {
                float dis = ptToLineDistance(points[i], {coef[0], coef[1]});
                if (fabs(dis) < threshold) {
                    P2 = i;
                }
            }
        }
    }

    Eigen::Vector2f slop = {0.f, -1.f/coef0[1]};  // Orthogonality
    Eigen::Vector2f coef_orthog;
    if (arrowType == 1) {
        coef_orthog = {points[term2].y() - slop[1] * points[term2].x(), slop[1]};
    } else if (arrowType == 2) {
        coef_orthog = {points[term1].y() - slop[1] * points[term1].x(), slop[1]};
    } else {
        return output;
    }

    if (P5 > P2) {
        for (int i=P2+1; i<P5; ++i) {
            float dis = ptToLineDistance(points[i], coef_orthog);
            if (fabs(dis) < threshold) {
                if (P3 == -1) {
                    P3 = i;
                }
                P4 = i;
            }
        }
        
    } else {
        for (int i=P2+1; i<points.size(); ++i) {
            float dis = ptToLineDistance(points[i], coef_orthog);
            if (fabs(dis) < threshold) {
                if (P3 == -1) {
                    P3 = i;
                }
                P4 = i;
            }
        } for (int i=0; i<P5; ++i) {
            float dis = ptToLineDistance(points[i], coef_orthog);
            if (fabs(dis) < threshold) {
                if (P3 == -1) {
                    P3 = i;
                }
                P4 = i;
            }
        }
    }

    if (P2 > 0) {
        output[2] = points[P2];
    }
    if (P3 > 0) {
        output[3] = points[P3];
    }
    if (P4 > 0) {
        output[4] = points[P4];
    }
    if (P5 > 0) {
        output[5] = points[P5];
    }


    // Refine P2/P3/P4/P5
    if (P1 == P2 || P5 == P6) { throw std::runtime_error("[Arrow Corner BUG] P1==P2 OR P5==P6"); };

    Eigen::Vector4f coef16_ = polyfit({{output[1], output[6]}}, 1);
    Coef coef16 = pt2ToCoef(output[1], output[6]);

    if (arrowType == 1) {
        // P2, P3
        refine2345(maxPt1, term2, m_sigma/2.f, coef0, coef16, output.begin()+2, coef_orthog, output.begin()+3);
        
        // P5, P4
        if (maxPt2 > term2) {
            refine2345(term2, maxPt2, m_sigma/2.f, coef0, coef16, output.begin()+5, coef_orthog, output.begin()+4);
        } else {
            refine2345(term2, maxPt2, m_sigma/2.f, coef0, coef16, output.begin()+5, coef_orthog, output.begin()+4);
        }

    } else if (arrowType == 2) {
        // P5, P4
        refine2345(term1, maxPt1, m_sigma/2.f, coef0, coef16, output.begin()+5, coef_orthog, output.begin()+4);

        // P2, P3
        if (term1 > maxPt2) {
            refine2345(maxPt2, term1, m_sigma/2.f, coef0, coef16, output.begin()+2, coef_orthog, output.begin()+3);
        } else {
            refine2345(maxPt2, term1, m_sigma/2.f, coef0, coef16, output.begin()+2, coef_orthog, output.begin()+3);
        }
    }

    return output;
}

ArrowLine ArrowCorner::fitLineByRansac(std::vector<Eigen::Vector2f>::const_iterator start, std::vector<Eigen::Vector2f>::const_iterator end, int iterations, int randNumbers, float sigma, int fitOrder)
{
    ArrowLine output;

    auto randN = [] (int max)->int {
        if (max > 0)
            return rand() % max;
        else
            return 0;
    };

    int n = end-start, outputIdx=0, loop=0;
    if (n < 2) return output;

    int maxScore=0;
    for (int k=0; k<iterations; ++k)
    {
        int score=0;

        // Generate sampPoints
        std::vector<Eigen::Vector2f> sampPoints;
        int rand1 = randN(n);
        sampPoints.push_back(*(start+rand1));
        int loop_i=0;
        while (sampPoints.size() < 2 && loop_i++ < 50)    //////////////////////
        {
            int rand2 = randN(n);

            if (rand1 != rand2) {
                sampPoints.push_back(*(start+rand2));
            }
        }
        // cout<<"SAMPPOINTS= ("<<sampPoints[0].x()<<","<<sampPoints[0].y()<<"), ("<<sampPoints[1].x()<<","<<sampPoints[1].y()<<"), len="<<sampPoints.size()<<endl;
        
        if (sampPoints.size() > 1)
        {
            // Fitting
            Eigen::Vector4f coef = polyfit({sampPoints}, fitOrder);

            // Check Err & Sigma
            // float deltaX = maxx - minx;
            float sumErr=0;
            for (std::vector<Eigen::Vector2f>::const_iterator it=start; it != end; ++it)
            {
                Eigen::Vector2f pt = *it;

                Coef coef1D = {coef[0], coef[1]};
                float err = ptToLineDistance(pt, coef1D);

                sumErr += err;
                if (err < sigma) {
                    ++score;
                }
            }

            // Compare Score & Update Data
            if (score > maxScore)
            {
                output.coef = {coef[0], coef[1]};
                output.ptsNumbers = score;
                maxScore = score;
            }
        }
    }

    return output;
}

float ArrowCorner::ptToLineDistance(const Eigen::Vector2f& pt, const Coef& lineCoef)
{
    float v = (pt.y() - lineCoef[1]*pt.x() - lineCoef[0]);
    float distance = v*v / (lineCoef[1] * lineCoef[1] + 1.f);

    return std::sqrt(distance);
}

Eigen::Vector2f ArrowCorner::linesToInterscep(const Coef& line1, const Coef& line2)
{
    float a1 = line1[1], c1 = line1[0];
    float a2 = line2[1], c2 = line2[0];

    return {(c1 - c2) / (a2 - a1), (a2*c1 - a1*c2) / (a2 - a1)};
}

float ArrowCorner::pointsToDist(const Eigen::Vector2f pt1, const Eigen::Vector2f pt2)
{
    float dist_sq;
    dist_sq = std::pow(pt1.x()-pt2.x(), 2) + std::pow(pt1.y()-pt2.y(), 2);
    return std::sqrt(dist_sq);
}

float ArrowCorner::coefToTheta(Coef coef1, Coef coef2)
{
    float m1=coef1[1], m2=coef2[1];
    return std::atan(abs((m1 - m2) / (1 + m1*m2)));
}

Coef ArrowCorner::periodsToCoef(int pos1, int pos2, const std::vector<Eigen::Vector2f>& points)
{
    if (pos1 == pos2) {throw std::runtime_error("pos1 cannot equal to pos2 !"); }

    std::vector<Eigen::Vector2f>::const_iterator it = points.begin();
    Eigen::Vector4f coef;

    if (pos2 > pos1) {
        std::vector<Eigen::Vector2f> period(it+pos1, it+pos2);
        coef = polyfit({period}, 1);
    } else {
        std::vector<Eigen::Vector2f> period1(it+pos1, it+points.size()-1);
        std::vector<Eigen::Vector2f> period2(it, it+pos2);
        coef = polyfit({period1, period2}, 1);
    }

    return {coef[0], coef[1]};
}

std::vector<Eigen::Vector2f> ArrowCorner::periodsToPoints(int pos1, int pos2, const std::vector<Eigen::Vector2f>& points)
{
    std::vector<Eigen::Vector2f>::const_iterator it = points.cbegin();
    std::vector<Eigen::Vector2f> pts;
    if (pos2 > pos1) {
        for (int i=pos1; i<pos2; ++i) {
            pts.push_back({(it+i)->x(), (it+i)->y()});
        }
    } else {
        for (int i=pos1; i<points.size(); ++i) {
            pts.push_back(*(it+i));
        } for (int i=0; i<pos2; ++i) {
            pts.push_back({(it+i)->x(), (it+i)->y()});

        }
    }

    return pts;
}

Coef ArrowCorner::ptSlopToCoef(Eigen::Vector2f pt, float slop)
{
    return {pt.y() - slop * pt.x(), slop};
}

Point ArrowCorner::pointsToCenter(Eigen::Vector2f pt1, Eigen::Vector2f pt2)
{
    return {(pt1.x()+pt2.x())/2.f, (pt1.y()+pt2.y())/2.f};
}

Coef ArrowCorner::pt2ToCoef(Eigen::Vector2f pt1, Eigen::Vector2f pt2)
{
    Eigen::Vector4f coef = polyfit({{pt1, pt2}}, 1);

    return {coef[0], coef[1]};
}

std::vector<cv::Point> ArrowCorner::inversePts(const std::vector<cv::Point>& pts)
{
    std::vector<cv::Point> output;
    for (int i=pts.size()-1; i>=0; --i)
    {
        output.push_back({pts[i].x, pts[i].y});
    }
    return output;
}

Eigen::Vector2f ArrowCorner::pt3(Eigen::Vector2f& inputPt) 
{
    float Y = (inputPt.x() - 512/2) * 0.0195; 
    float X = ((512-1) - inputPt.y() -512/2) * 0.0195;
    return {X, Y};
}

std::vector<Eigen::Vector2f> ArrowCorner::arrowTo3D_2(const std::vector<cv::Point>& arrow2D, int threshold)
{
    std::vector<Eigen::Vector2f> out;
    if (arrow2D.size() > threshold)
    {
        for (auto pt : arrow2D) {
            Eigen::Vector2f ptIn = {float(pt.x), float(pt.y)};
            Eigen::Vector2f ptOut = pt3(ptIn);
            out.push_back(ptOut);
        }
    }

    return out;
}

std::vector<std::vector<Eigen::Vector2f>> ArrowCorner::arrowsTo3D_2(const std::vector<std::vector<Eigen::Vector2f>>& arrows2D, int threshold)
{
    std::vector<std::vector<Eigen::Vector2f>> out;
    for (auto arrow : arrows2D) {
        if (arrow.size() > threshold)
        {
            std::vector<Eigen::Vector2f> tempCurve;
            for (auto pt : arrow) {
                Eigen::Vector2f ptOut = pt3(pt);
                tempCurve.push_back(ptOut);
            }
            out.push_back(tempCurve);
        }
    }

    return out;
}

cv::Point2f ArrowCorner::pt3_inv_cv(const cv::Point2f& inputPt) 
{
    float x = inputPt.y/0.0195 + 512/2;
    float y = (512-1) - (512/2) - inputPt.x / 0.0195;
    return {x, y};
}

std::vector<cv::Point2f> ArrowCorner::arrowTo2D_cv(const std::vector<Eigen::Vector2f>& inputPts3d)
{
    std::vector<cv::Point2f> output;
    for (auto pt : inputPts3d)
    {
        output.push_back(pt3_inv_cv({pt.x(), pt.y()}));
    }
    return output;
}

Eigen::Vector4f ArrowCorner::polyfit(const std::vector<std::vector<Eigen::Vector2f>>& lanesGroup, int order)
{
    auto countPixel =[](const std::vector<std::vector<Eigen::Vector2f>>& lanes)
    {
        int size = 0;
        for (auto& lane: lanes) {
            size += lane.size();
        }
        return size;
    };
    
    int lanePixSize = countPixel(lanesGroup);
    Eigen::MatrixXd T(lanePixSize, order + 1);
    Eigen::VectorXd V(lanePixSize, 1);
    Eigen::VectorXd result;
    Eigen::Vector4f coefs;

    int idx;

    // init T
    idx = 0;
    for (size_t i=0; i<lanesGroup.size(); ++i) {
        for (size_t j=0; j<lanesGroup[i].size(); ++j) {
            for (int k=0; k<order+1; ++k) {
                T(idx, k) = pow(lanesGroup[i][j].x(), k);
            }
            ++idx;
        }
    }

    // init V
    idx = 0;
    for (size_t i=0; i<lanesGroup.size(); ++i) {
        for (size_t j=0; j<lanesGroup[i].size(); ++j) {
            V(idx, 0) = lanesGroup[i][j].y();
            idx += 1;
        }
    }

    result = T.householderQr().solve(V);
    for (int k=0; k<order+1; ++k) {
        coefs[k] = result[k];
    }
    return coefs;
}

std::vector<cv::Point2f> ArrowCorner::eigen2cvPoint(std::vector<Eigen::Vector2f>& input)
{
    std::vector<cv::Point2f> output;

    for (auto pt : input) {
        output.push_back({float(pt.x()), float(pt.y())});
    }
    return output;
}
