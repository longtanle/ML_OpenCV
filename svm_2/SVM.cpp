#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

const Scalar WHITE_COLOR = Scalar(255,255,255);
const string winName = "points";
const int testStep = 5;

Mat img, imgDst;
RNG rng;

/** To store the position of points in image **/
vector<Point>  trainedPoints;
/** To store the color of points in image **/
vector<int>    trainedPointsMarkers;
/** 2 classes Red and Green **/
const int MAX_CLASSES = 2;
vector<Vec3b>  classColors(MAX_CLASSES);

int currentClass = 0;
vector<int> classCounters(MAX_CLASSES);

/** Read pixel value from image **/
void memory_pattern(ofstream &outfile, cv::Mat image)
{
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            outfile << &image.at<int>(r,c) << ",";
        }
        outfile << endl;
    }
}

/** Using Mouse Click to create points in image **/
static void on_mouse( int event, int x, int y, int /*flags*/, void* )
{
    if( img.empty() )
        return;

    int updateFlag = 0;

    if( event == EVENT_LBUTTONUP )
    {
        trainedPoints.push_back( Point(x,y) );
        trainedPointsMarkers.push_back( currentClass );
        classCounters[currentClass]++;
        updateFlag = true;
    }

    //draw
    if( updateFlag )
    {
        img = Scalar::all(0);

        // draw points
        for( size_t i = 0; i < trainedPoints.size(); i++ )
        {
            Vec3b c = classColors[trainedPointsMarkers[i]];
            circle( img, trainedPoints[i], 5, Scalar(c), -1 );
        }

        imshow( winName, img );
   }
}

/** Convert train data format **/
static Mat prepare_train_samples(const vector<Point>& pts)
{
    Mat samples;
    Mat(pts).reshape(1, (int)pts.size()).convertTo(samples, CV_32F);
    return samples;
}

/** Prepare Train Data **/
static Ptr<TrainData> prepare_train_data()
{
    Mat samples = prepare_train_samples(trainedPoints);
    return TrainData::create(samples, ROW_SAMPLE, Mat(trainedPointsMarkers));
}

/** Pain the result image to classify the points**/
static void predict_and_paint(const Ptr<StatModel>& model, Mat& dst)
{
    // Defines both the depth of each element and the number of channels
    Mat testSample( 1, 2, CV_32FC1 );

    for( int y = 0; y < img.rows; y += testStep )
    {
        for( int x = 0; x < img.cols; x += testStep )
        {
            testSample.at<float>(0) = (float)x;
            testSample.at<float>(1) = (float)y;

            int response = (int)model->predict( testSample );
            dst.at<Vec3b>(y, x) = classColors[response];
        }
    }
}

/** Apply Support Vector Machine to classify the color of points **/
static void find_decision_boundary_SVM( double C )
{
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::POLY); //SVM::LINEAR;
    svm->setDegree(0.5);
    svm->setGamma(1);
    svm->setCoef0(1);
    svm->setNu(0.5);
    svm->setP(0);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
    svm->setC(C);
    svm->train(prepare_train_data());
    predict_and_paint(svm, imgDst);

    Mat sv = svm->getSupportVectors();
    for( int i = 0; i < sv.rows; i++ )
    {
        const float* supportVector = sv.ptr<float>(i);
        circle( imgDst, Point(saturate_cast<int>(supportVector[0]),saturate_cast<int>(supportVector[1])), 5, Scalar(255,255,255), -1 );
    }
}


int main()
{
    cout << "Press:" << endl
         << "  Key 'ESC' - To exit program" << endl
         << "  Key '0' or '1' - To change color of points ('0': green; '1': red)" << endl
         << "  Left mouse button - To add new point;" << endl
         << "  Key 'r' - To run the ML model;" << endl
         << "  Key 'i' - To init (clear) the data." << endl << endl;

    cv::namedWindow( "points", 1 );
    img.create( 480, 640, CV_8UC3 );
    imgDst.create( 480, 640, CV_8UC3 );

    imshow( "points", img );
    setMouseCallback( "points", on_mouse );

    classColors[0] = Vec3b(0, 255, 0);
    classColors[1] = Vec3b(0, 0, 255);

    for(;;)
    {
        char key = (char)waitKey();

        if( key == 27 ) break;

        if( key == 'i' ) // init
        {
            img = Scalar::all(0);

            trainedPoints.clear();
            trainedPointsMarkers.clear();
            classCounters.assign(MAX_CLASSES, 0);

            imshow( winName, img );
        }

        if( key == '0' || key == '1' )
        {
            currentClass = key - '0';
        }

        if( key == 'r' ) // run
        {
            double minVal = 0;
            minMaxLoc(classCounters, &minVal, 0, 0, 0);
            if( minVal == 0 )
            {
                printf("each class should have at least 1 point\n");
                continue;
            }
            img.copyTo( imgDst );

            find_decision_boundary_SVM( 1 );
            imshow( "classificationSVM1", imgDst );

            find_decision_boundary_SVM( 10 );
            imshow( "classificationSVM2", imgDst );
        }
    }

    return 0;
}
