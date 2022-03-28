// Edge Detection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <math.h>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

Mat preProcessing(Mat image)
{
    Mat blurImage;
    GaussianBlur(image, blurImage, Size(3, 3), 1);

    Mat grayImage;
    cvtColor(blurImage, grayImage, COLOR_BGR2GRAY);

    blurImage.release();
    return grayImage;
}

int xGradientSobel(Mat image, int x, int y)
{
    return -image.at<uchar>(x - 1, y - 1) - 2 * image.at<uchar>(x - 1, y + 1)
        - image.at<uchar>(x - 1, y + 1) + image.at<uchar>(x + 1, y - 1)
        + 2 * image.at<uchar>(x + 1, y) + image.at<uchar>(x + 1, y + 1);
}

int yGradientSobel(Mat image, int x, int y)
{
    return image.at<uchar>(x - 1, y - 1) + 2 * image.at<uchar>(x, y - 1)
        + image.at<uchar>(x + 1, y - 1) - image.at<uchar>(x - 1, y + 1)
        - 2 * image.at<uchar>(x, y + 1) - image.at<uchar>(x + 1, y + 1);
}

int xGradientPrewitt(Mat image, int x, int y)
{
    return -image.at<uchar>(x - 1, y - 1) - image.at<uchar>(x - 1, y)
        - image.at<uchar>(x - 1, y + 1) + image.at<uchar>(x + 1, y - 1)
        + image.at<uchar>(x + 1, y) + image.at<uchar>(x + 1, y + 1);
}

int yGradientPrewitt(Mat image, int x, int y)
{
    return -image.at<uchar>(x - 1, y - 1) - image.at<uchar>(x, y - 1)
        - image.at<uchar>(x + 1, y - 1) + image.at<uchar>(x - 1, y + 1)
        + image.at<uchar>(x, y + 1) + image.at<uchar>(x + 1, y + 1);
}

int gradient(Mat image, int x, int y, string type)
{
    if (x < 1 || y < 1)
        return 0;
    else if (x == image.rows - 1 || y == image.cols - 1)
        return 0;

    int gx, gy, g;
    g = gx = gy = 0;
    if (type == "Sobel" || type == "sobel")
    {
        gx = xGradientSobel(image, x, y);
        gy = yGradientSobel(image, x, y);
    }
    else if (type == "Prewitt" || type == "prewitt")
    {
        gx = xGradientPrewitt(image, x, y);
        gy = yGradientPrewitt(image, x, y);
    }
    else if (type == "Laplace" || type == "laplace")
    {
        return image.at<uchar>(x, y - 1) + image.at<uchar>(x - 1, y) + image.at<uchar>(x + 1, y)
            + image.at<uchar>(x, y + 1) - 4 * image.at<uchar>(x, y);
    }

    g = sqrt(gx * gx + gy * gy);
    return g;
}

int angleDirection(double a)
{
    double min = 0;
    if (a <= 0)
        a += 180;
    else if (a >= 180)
        a -= 180;
    double angleList[] = { 0, 45,90, 135, 180 };
    for (int i = 0; i < 5; ++i)
        if (abs(angleList[i] - a) < abs(min - a))
            min = angleList[i];
    if (min == 180)
        return 0;
    return int(min);
}

int isEdge(Mat image, int x, int y, int direction)
{
    int g, g1, g2;
    g1 = g2 = g = 0;

    g = gradient(image, x, y, "sobel");
    if (direction == 0)
    {
        g1 = gradient(image, x - 1, y, "sobel");
        g2 = gradient(image, x + 1, y, "sobel");
    }
    else if (direction == 45)
    {
        g1 = gradient(image, x - 1, y - 1, "sobel");
        g2 = gradient(image, x + 1, y + 1, "sobel");
    }
    else if (direction == 90)
    {
        g1 = gradient(image, x, y - 1, "sobel");
        g2 = gradient(image, x, y + 1, "sobel");
    }
    else if (direction == 135)
    {
        g1 = gradient(image, x + 1, y - 1, "sobel");
        g2 = gradient(image, x - 1, y + 1, "sobel");
    }
    else {
        cout << "Loi direction khong hop le\n";
        return 0;
    }
    
    if (g > g1 && g > g2)
        return 1;
    return 0;
}

bool isNeighborOfEdge(Mat image, int x, int y, int highThreshold)
{
    int horizontalSearch[] = { -1, 0, 1 };
    int verticalSearch[] = { -1,0,1 };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (gradient(image, horizontalSearch[i], verticalSearch[j], "sobel") >= highThreshold)
                return true;
    return false;
}

//*****************************************************************************//

int detectedBySobel(Mat src, Mat des, int x, int y)
{
    int gx, gy, g;
    
    if (x == 1 && y == 0)
    {
        // Horizontal 
        for (int i = 1; i < src.rows - 1; ++i)
            for (int j = 1; j < src.cols - 1; ++j)
            {
                gx = xGradientSobel(src, i, j);
                des.at<uchar>(i, j) = gx;
            }
        namedWindow("Sobel edge detection horizontal", WINDOW_AUTOSIZE);
        imshow("Sobel edge detection horizontal", des);
        waitKey(0);
    }
    else if (x == 0 && y == 1)
    {
        // Vertical
        for (int i = 1; i < src.rows - 1; ++i)
            for (int j = 1; j < src.cols - 1; ++j)
            {
                gy = yGradientSobel(src, i, j);
                des.at<uchar>(i, j) = gy;
            }
        namedWindow("Sobel edge detection vertical", WINDOW_AUTOSIZE);
        imshow("Sobel edge detection vertical", des);
        waitKey(0);
    }
    else if (x == 1 && y == 1)
    {
        // Magnitude
        for (int i = 1; i < src.rows - 1; i++)
        {
            for (int j = 1; j < src.cols - 1; j++)
            {
                g = gradient(src, i, j, "sobel");
                des.at<uchar>(i, j) = g;
            }
        }
        namedWindow("Sobel edge detection horizontal and vertical", WINDOW_AUTOSIZE);
        imshow("Sobel edge detection horizontal and vertical", des);
        waitKey(0);
    }

    return 1;
}

int detectedByPrewitt(Mat src, Mat des, int x, int y)
{
    int gx, gy, g;
    des = src.clone();

    if (x == 1 && y == 0)
    {
        // Horizontal 
        for (int i = 1; i < src.rows - 1; ++i)
            for (int j = 1; j < src.cols - 1; ++j)
            {
                gx = xGradientPrewitt(src, i, j);
                des.at<uchar>(i, j) = gx;
            }
        namedWindow("Prewitt edge detection horizontal", WINDOW_AUTOSIZE);
        imshow("Prewitt edge detection horizontal", des);
        waitKey(0);
    }
    else if (x == 0 && y == 1)
    {
        // Vertical
        for (int i = 1; i < src.rows - 1; ++i)
            for (int j = 1; j < src.cols - 1; ++j)
            {
                gy = yGradientPrewitt(src, i, j);
                des.at<uchar>(i, j) = gy;
            }
        namedWindow("Prewitt edge detection vertical", WINDOW_AUTOSIZE);
        imshow("Prewitt edge detection vertical", des);
        waitKey(0);
    }
    else if (x == 1 && y == 1)
    {
        // Magnitude
        for (int i = 1; i < src.rows - 1; i++)
        {
            for (int j = 1; j < src.cols - 1; j++)
            {
                g = gradient(src, i, j, "prewitt");
                des.at<uchar>(i, j) = g;
            }
        }
        namedWindow("Prewitt edge detection horizontal and vertical", WINDOW_AUTOSIZE);
        imshow("Prewitt edge detection horizontal and vertical", des);
        waitKey(0);
    }

    return 1;
}

int detectedByLaplace(Mat src, Mat des)
{
    int g;
    for(int i = 1; i< src.rows-1;++i)
        for (int j = 1; j < src.cols - 1; ++j)
        {
            g = gradient(src, i, j, "laplace");
            des.at<uchar>(i, j) = g;
        }
    namedWindow("Edge detection by Laplace", WINDOW_AUTOSIZE);
    imshow("Edge detection by Laplace", des);
    waitKey(0);
    return 1;
}

int detectedByCanny(Mat src, Mat des)
{
    int gx, gy, g;
    double tan;
    int direction;
    int maxGradient = -1;
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            // Sobel filter
            gx = xGradientSobel(src, i, j);
            gy = yGradientSobel(src, i, j);
            g = sqrt(gx * gx + gy * gy);
            if (g > maxGradient)
                maxGradient = g;
            tan = atan2(gy * 1.0, gx * 1.0) * 180 / 3.14159265;
            direction = angleDirection(tan);
            // Non-maximum suppresion
            if (isEdge(src, i, j, direction) == 1)
                des.at<uchar>(i, j) = g;
            else des.at<uchar>(i, j) = 0;
        }
    }

    // Double Threshold
    int highThreshold = maxGradient * 0.12;
    int lowThreshold = maxGradient * 0.08;

    for(int i =1; i< src.rows; ++i)
        for (int j = 1; j < src.cols; ++j)
        {
            g = gradient(src, i, j, "sobel");
            if (g >= highThreshold)
                des.at<uchar>(i, j) = 255;
            else if (g >= lowThreshold)
            {
                if (isNeighborOfEdge(src, i, j, highThreshold) == 1)
                    des.at<uchar>(i, j) = g;
                else des.at<uchar>(i, j) = 0;
            }
            else
                des.at<uchar>(i, j) = 0;
        }

    namedWindow("Canny edge detection", WINDOW_AUTOSIZE);
    imshow("Canny edge detection", des);
    waitKey(0);
    return 1;
}

int main(int argc, char** argv)
{

    if (argc < 3)
    {
        cout << "Tham so truyen vao khong du\n";
        return -1;
    }
    else if (argc == 3)
    {
  
        Mat image = imread(argv[1], IMREAD_COLOR);
        if (!image.data)
        {
            cout << "Loi: khong the mo anh\n";
            return -1;
        }
        Mat src = preProcessing(image);

        Mat des = src.clone();

        if (strcmp(argv[2], "sobel") == 0)
            int a = detectedBySobel(src, des, 1, 1);
        else if (strcmp(argv[2], "prewitt") == 0)
            int b = detectedByPrewitt(src, des, 1, 1);
        else if (strcmp(argv[2], "laplace") == 0)
            int c = detectedByCanny(src, des);
        else if (strcmp(argv[2], "canny") == 0)
            int d = detectedByLaplace(src, des);
        else cout << "Cu phap cho viec khoi chay chuong trinh khong hop le\n";

        /*Sobel(src, des, CV_64F, 1, 1);

        namedWindow("display", WINDOW_AUTOSIZE);
        imshow("display", des);
        waitKey(0);*/

        image.release();
        src.release();
        des.release();
    }
    
}