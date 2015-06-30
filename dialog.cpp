#include "dialog.h"
#include "ui_dialog.h"
#include <QMessageBox>
#include <QDebug>
#include <cv.h>
#include <opencv/highgui.h>
#include <cmath>
#include <QString>
#include <QVector>

/////////////////////////////////////////////////////////////////////////////
Dialog::Dialog(QWidget *parent) : QDialog(parent),ui(new Ui::Dialog)
{
    ui->setupUi(this);

    refreshTime = 20;

    capWebcam.open(0);
    capWebcam.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    capWebcam.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    if(!capWebcam.isOpened())
    {
        QMessageBox::information(this, "Error accessing the webcam", "There was an error when trying to acces the webcam");
        return;
    }

    tmrTimer = new QTimer(this);
    connect(tmrTimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->ycbcrCheckbox, SIGNAL(clicked()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->hsvCheckBox, SIGNAL(clicked()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->rgbCheckbox, SIGNAL(clicked()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->lowCannyThresh, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->highCannyThresh, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->eyesThreshSlider, SIGNAL(sliderReleased()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->mouthThreshSlider, SIGNAL(sliderReleased()), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->eyesThreshSlider, SIGNAL(sliderMoved(int)), ui->eyesThreshLcd, SLOT(display(int)));
    connect(ui->mouthThreshSlider, SIGNAL(sliderMoved(int)), ui->mouthThreshLcd, SLOT(display(int)));
    connect(ui->kernelSizeSpin, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->skinIt, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->eyesIt, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->mouthIt, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->blurIntensity, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->deviceNumber, SIGNAL(valueChanged(int)), this, SLOT(changeChannel()));
    connect(ui->cannyDilate, SIGNAL(valueChanged(int)), this, SLOT(processFrameAndUpdateGUI()));
    connect(ui->hide, SIGNAL(clicked()), this, SLOT(changeSize()));
    connect(ui->quit, SIGNAL(clicked()), qApp, SLOT(quit()));

    tmrTimer->start(refreshTime);
}
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
Dialog::~Dialog()
{
    delete ui;
}
/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
void Dialog::processFrameAndUpdateGUI(){

    capWebcam.read(matOriginal);

    if(matOriginal.empty())return;

    cv::Mat str = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ui->kernelSizeSpin->value(),ui->kernelSizeSpin->value()), cv::Point(1,1));

    float a = 0;
    int centX = 0;
    int centY = 0;

    float ratioOfHead = 0;
    double heightOfHead = 0;
    double widthOfHead = 0;
    bool faceDetected = false;

    cv::Mat3b blurredOriginal = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);

    cv::Mat3b bw        =  cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat3b ycbcrskin =  cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat3b rgbskin   =  cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat3b hsvSkin   =  cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);

    cv::Mat skinEdges;

    cv::medianBlur(matOriginal, blurredOriginal,ui->blurIntensity->value());

    cv::cvtColor(blurredOriginal, hsvSkin, CV_BGR2HSV);
    cv::cvtColor(blurredOriginal, ycbcrskin, CV_BGR2YCrCb);
    cv::cvtColor(blurredOriginal, rgbskin, CV_BGR2RGB);

    cv::Canny(blurredOriginal, skinEdges, ui->lowCannyThresh->value(),ui->highCannyThresh->value());
    cv::dilate(skinEdges, skinEdges, cv::Mat1b(3,3,1), cv::Point(1,1), ui->cannyDilate->value());

    if(ui->hsvCheckBox->checkState()){
        for(int r=0; r<bw.rows; ++r){
            for(int c=0; c<bw.cols; ++c){

                int h = hsvSkin(r,c)[0];
                int s = hsvSkin(r,c)[1];
                int v = hsvSkin(r,c)[2];

                if( (h>=0) && (h <=50) && (s>=58) && (s<=174))
                {bw(r,c)[0] = 255; bw(r,c)[1] = 255; bw(r,c)[2] = 255;}
                else{bw(r,c)[0] = 0; bw(r,c)[1] = 0; bw(r,c)[2] = 0;}
            }
        }
    }


    if(ui->ycbcrCheckbox->checkState()){
        for(int r=0; r<bw.rows; ++r){
            for(int c=0; c<bw.cols; ++c){

                int y = ycbcrskin(r,c)[0];
                int cb = ycbcrskin(r,c)[2];
                int cr = ycbcrskin(r,c)[1];

                if((y > 80) && ((cr >=  133) && (cr <=  173)) && ((cb >=  80) && (cb <=  120)))
                {bw(r,c)[0] = 255; bw(r,c)[1] = 255; bw(r,c)[2] = 255;}
                else{bw(r,c)[0] = 0; bw(r,c)[1] = 0; bw(r,c)[2] = 0;}
            }
        }
    }

    if(ui->rgbCheckbox->checkState()){
        for(int r=0; r<bw.rows; ++r){
            for(int c=0; c<bw.cols; ++c){

                int r1 = rgbskin(r,c)[0];
                int g = rgbskin(r,c)[1];
                int b = rgbskin(r,c)[2];

                int diffAbs = abs(r1 - g);
                float maxCol = fmax(r1, fmax(g,b));
                float minCol = fmin(r1, fmin(g,b));

                bool daylight = (r1 > 95 && g > 40 && b > 20) && ((maxCol - minCol) > 15) && (diffAbs > 15 && r1 > g && r1 > b);
                bool flashlight = (r1 > 220) && (g > 210) && (b > 170) && (diffAbs <= 15) && (r1 > b) && (g > b);

                if(daylight || flashlight)
                {bw(r,c)[0] = 255; bw(r,c)[1] = 255; bw(r,c)[2] = 255;}
                else{bw(r,c)[0] = 0; bw(r,c)[1] = 0; bw(r,c)[2] = 0;}

            }
        }
    }

    cv::Mat1b bw2, bw3;
    cv::cvtColor(bw, bw2, CV_BGR2GRAY);
    cv::threshold(bw2, bw2, 60, 255, CV_THRESH_BINARY);

    cv::morphologyEx(bw2, bw2, CV_MOP_ERODE, str, cv::Point(1,1), ui->skinIt->value());
    cv::morphologyEx(bw2, bw2, CV_MOP_DILATE, str, cv::Point(1,1), ui->skinIt->value());

    cv::bitwise_not(skinEdges, skinEdges);
    cv::bitwise_and(skinEdges, bw2, bw3);


    cv::Mat label = cv::Mat::zeros(bw2.size(), CV_8UC3);
    cv::Mat stats, centroids;

    int nbLabels = cv::connectedComponentsWithStats(bw3, label, stats, centroids, 4, CV_32S);
    float x = 0, y = 0, x1 = 0, y1 = 0;

    for (int i = 1; i < nbLabels; ++i) {
        a = stats.at<int>(i, cv::CC_STAT_AREA);

        if(a > ui->lowFaceSurface->value() && a < ui->highFaceSurface->value())
        {
            centX = centroids.at<double>(i,0);
            centY = centroids.at<double>(i,1);

            widthOfHead = stats.at<int>(i, cv::CC_STAT_WIDTH);
            heightOfHead = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            if(widthOfHead < heightOfHead)
                ratioOfHead = widthOfHead / heightOfHead;
            else
                ratioOfHead = heightOfHead / widthOfHead;

            if(ratioOfHead >= 0.5 && ratioOfHead <= 1.1){
                x = stats.at<int>(i, cv::CC_STAT_LEFT);
                y = stats.at<int>(i, cv::CC_STAT_TOP);
                x1 = stats.at<int>(i, cv::CC_STAT_WIDTH) + x;
                y1 = stats.at<int>(i, cv::CC_STAT_HEIGHT) + y;
                faceDetected = true;
            }

            break;
        }
    }

    ///
    /// \brief Creation of the eye map
    ///

    cv::Mat3b eyemapC = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat eyemapCeq= cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);

    cv::Mat3b luma = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat3b lumaEroded, lumaDilated;
    cv::Mat3b lumaFinal = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat lumaFeq= cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);

    cv::Mat3b eyeMapFinal = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);

    cv::Mat labelEyes = cv::Mat::zeros(bw2.size(), CV_8UC3);
    cv::Mat statsEyes, centroidsEyes, eyesToLabel;

    cv::cvtColor(matOriginal, eyemapC, CV_BGR2YCrCb);

    for(int r=0; r<eyemapC.rows; ++r){
        for(int c=0; c<eyemapC.cols; ++c){

            int y = eyemapC(r,c)[0];
            int cr = eyemapC(r,c)[1];
            int cb = eyemapC(r,c)[2];

            int crSqr = ((double)(cb * cb) / (65025.0)) * 255.0;
            int crNeg = (((double)(255-cr) * (255-cr)) / (65025.0)) * 255.0;
            int crcbRatio = ((((double) cb / (double) cr) * 512.0) > 255) ? 255 : (((double) cb / (double) cr) * 512.0);

            double emc = 0;

            emc = (crSqr + crNeg + crcbRatio) / 3.0;
            //emc = 2.5 * cb - 1.4 * cr;
            if(emc < 0) emc = 0; else if (emc > 255) emc = 255;

            eyemapC(r,c)[0] = emc;
            eyemapC(r,c)[1] = emc;
            eyemapC(r,c)[2] = emc;

            luma(r,c)[0] = y;
            luma(r,c)[1] = y;
            luma(r,c)[2] = y;
        }
    }

    cv::cvtColor(eyemapC, eyemapC, CV_YCrCb2BGR);
    cv::cvtColor(eyemapC, eyemapCeq, CV_BGR2GRAY);
    cv::equalizeHist(eyemapCeq, eyemapCeq);

    cv::morphologyEx(luma, lumaDilated, CV_MOP_DILATE, str, cv::Point(1,1), ui->eyesIt->value());
    cv::morphologyEx(luma, lumaEroded, CV_MOP_ERODE, str, cv::Point(1,1), ui->eyesIt->value());

    for(int r=y; r<y1; ++r){
        for(int c=x; c<x1; ++c){

            int dil = lumaDilated(r,c)[0];
            int ero = lumaEroded(r,c)[0];

            double val = (double) dil / (ero + 255) ;
            val *= 255;
            val = (val > 255) ? 255 : val;

            lumaFinal(r,c)[0] = val; lumaFinal(r,c)[1] = val; lumaFinal(r,c)[2] = val;
        }
    }


    cv::cvtColor(lumaFinal, lumaFeq, CV_BGR2GRAY);
    cv::equalizeHist(lumaFeq,lumaFeq);

    for(int r= y + (y1 - y) * 0.2; r< y + (y1 - y) * 0.6; ++r){
        for(int c= ( x + 10); c< (x1-10); ++c){

            double value = 0;

            int ec = eyemapCeq.at<uchar>(r,c);
            int el = lumaFeq.at<uchar>(r,c);
            int sk = bw2.at<uchar>(r,c);

                value = ((ec * el) / 65025.0) * 255.0;

                if(value < ui->eyesThreshSlider->value())
                {eyeMapFinal(r,c)[0] = 0; eyeMapFinal(r,c)[1] = 0; eyeMapFinal(r,c)[2] = 0;}
                else
                {eyeMapFinal(r,c)[0] = 255; eyeMapFinal(r,c)[1] = 255; eyeMapFinal(r,c)[2] = 255;}
        }
    }

    //cv::rectangle(matOriginal, cv::Point(x+10, y + (y1 - y) * 0.20), cv::Point(x1-10, y + (y1 - y) * 0.6), cv::Scalar(128,128,0), 2);

    cv::Mat eyeMapFinalG;
    cv::cvtColor(eyeMapFinal, eyeMapFinalG, CV_BGR2GRAY);
    cv::morphologyEx(eyeMapFinalG, eyeMapFinalG, CV_MOP_ERODE, cv::Mat1b(3,3,1), cv::Point(1,1), 1);
    //cv::morphologyEx(eyeMapFinalG, eyeMapFinalG, CV_MOP_DILATE, cv::Mat1b(3,3,1), cv::Point(1,1), 1);
    cv::medianBlur(eyeMapFinalG, eyeMapFinalG, 3);
    //cv::threshold(eyeMapFinalG, eyeMapFinalG, 68, 255, CV_THRESH_BINARY);

    cv::cvtColor(eyeMapFinal, eyesToLabel, CV_BGR2GRAY);
    cv::dilate(eyesToLabel, eyesToLabel, str);

    int nbLabelsEyes = cv::connectedComponentsWithStats(eyesToLabel, labelEyes, statsEyes, centroidsEyes, 8, CV_32S);
    QVector<cv::Point> centers;

    for (int i = 1; i < nbLabelsEyes; ++i) {
        float eyesSurface = statsEyes.at<int>(i, cv::CC_STAT_AREA);

        if(eyesSurface > ui->lowEyeSurface->value() && eyesSurface < ui->highEyeSurface->value()){
            cv::Point center(centroidsEyes.at<double>(i,0), centroidsEyes.at<double>(i,1));
            centers.push_back(center);
        }
    }

    ///
    /// \brief Creation of th eye map
    ///


    /// ////////////////////////////////////////////////////////////
    /// Test ellipses donc peut supprimer tout ca quand tu veux ///
    /// //////////////////////////////////////////////////////////
    ///
    ///
    ///
    /// /// ////////////////////////////////////////////////////////
    /// Test ellipses donc peut supprimer tout ca quand tu veux ///
    /// //////////////////////////////////////////////////////////

//    std::vector<std::vector<cv::Point> > contours;
//    std::vector<cv::Vec4i> hierarchy;

//    cv::RNG rng(12345);

//    //Find contours
//    cv::findContours(eyeMapFinalG, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));

//    std::vector<cv::RotatedRect> minRect(contours.size());
//    std::vector<cv::RotatedRect> minEllipse(contours.size());

//    for (int i = 0; i < contours.size(); ++i) {
//        minRect[i] = cv::minAreaRect(cv::Mat(contours[i]));
//        if(contours[i].size() > 15)
//        {minEllipse[i] = cv::fitEllipse(cv::Mat(contours[i]));}
//    }

//    cv::Mat drawing = cv::Mat::zeros(240,320, CV_8UC3);
//    for (int i = 0; i < contours.size(); ++i) {
//        cv::Scalar color = cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
////      cv::drawContours(drawing, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
//        cv::ellipse(drawing, minEllipse[i], color, 2,8);
////      cv::Point2f rect_points[4]; minRect[i].points(rect_points);
////        for (int j = 0; j < 4; ++j) {
////            cv::line(drawing, rect_points[j], rect_points[(j+1) % 4], color, 1, 8);
////        }
//    }



    /// ////////////////////////////////////////////////////////////
    /// Test ellipses donc peut supprimer tout ca quand tu veux ///
    /// //////////////////////////////////////////////////////////
    ///
    ///
    ///
    /// ////////////////////////////////////////////////////////////
    /// Test ellipses donc peut supprimer tout ca quand tu veux ///
    /// //////////////////////////////////////////////////////////


    for (int j = 0; j < centers.size(); ++j) {
            cv::circle(matOriginal, centers[j], 8, cv::Scalar(0,255,0), 2);
        }

    ///////////////////////////////////////
    /// \brief Creation de la mouth map
    ///////////////////////////////////////

    cv::Mat3b mouthMap = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat3b mouthMapFinal = cv::Mat::zeros(matOriginal.rows, matOriginal.cols, CV_8UC1);
    cv::Mat labelMouth = cv::Mat::zeros(bw2.size(), CV_8UC3);
    cv::Mat statsMouth, centroidsMouth;
    cv::Mat mouthToLabel;

    cv::cvtColor(matOriginal, mouthMap, CV_BGR2YCrCb);

    double avgCr = 0;
    double avgCbCr = 0;
    double faceArea = 0;
    double mouthMapAvg = 0;
    double mouthMapValue = 0;

    for(int r=y; r<y1; ++r){
        for(int c=x; c<x1; ++c){

            double cr = mouthMap(r,c)[1] / 255.0;
            double cb = mouthMap(r,c)[2] / 255.0;

            avgCr = (cr * cr) + avgCr;
            avgCbCr = (cr / cb) + avgCbCr;

        }
    }

    faceArea = (y1 - y) * (x1 - x);
    mouthMapAvg = 0.95 * ((avgCr / faceArea) / (avgCbCr / faceArea));

    for(int r=y; r<y1; ++r){
        for(int c=x; c<x1; ++c){

            double cr = mouthMap(r,c)[1] / 255.0;
            double cb = mouthMap(r,c)[2] / 255.0;

            mouthMapValue = ((cr * cr) * pow(((cr * cr) - mouthMapAvg * (cr / cb)), 2)) * 255555.0;

            mouthMapValue = (mouthMapValue > 255) ? 255 : (mouthMapValue < ui->mouthThreshSlider->value()) ? 0 : mouthMapValue;

            mouthMapFinal(r,c)[0] = mouthMapValue;
            mouthMapFinal(r,c)[1] = mouthMapValue;
            mouthMapFinal(r,c)[2] = mouthMapValue;

        }
    }

    cv::medianBlur(mouthMapFinal, mouthMapFinal, (ui->blurIntensity->value() - 2));
    cv::morphologyEx(mouthMapFinal, mouthMapFinal, CV_MOP_ERODE, str, cv::Point(1, 1), ui->mouthIt->value());
    cv::morphologyEx(mouthMapFinal, mouthMapFinal, CV_MOP_DILATE, cv::Mat1b(3,3,1), cv::Point(1, 1), 5);

    cv::cvtColor(mouthMapFinal, mouthToLabel, CV_BGR2GRAY);
    cv::threshold(mouthToLabel, mouthToLabel, 10, 255, cv::THRESH_BINARY);

    cv::morphologyEx(mouthToLabel, mouthToLabel, CV_MOP_DILATE, cv::Mat1b(3,3,1), cv::Point(1, 1), 2);

    int nbLabelsMouth = cv::connectedComponentsWithStats(mouthToLabel, labelMouth, statsMouth, centroidsMouth, 8, CV_32S);
    int xM = -10, x1M= 0, yM= -10, y1M= 0;

    for (int i = 1; i < nbLabelsMouth; ++i) {

        float mouthSurface = statsMouth.at<int>(i, cv::CC_STAT_AREA);

        if(mouthSurface > ui->lowMouthSurface->value() && mouthSurface < ui->highMouthSurface->value()){

              xM = centroidsMouth.at<double>(i,0);
              yM = centroidsMouth.at<double>(i,1);
//            xM  = statsMouth.at<int>(i, cv::CC_STAT_LEFT);
//            yM  = statsMouth.at<int>(i, cv::CC_STAT_TOP);

//            x1M = statsMouth.at<int>(i, cv::CC_STAT_WIDTH) + xM;
//            y1M = statsMouth.at<int>(i, cv::CC_STAT_HEIGHT) + yM;

            break;

            }
        }

    cv::circle(matOriginal, cv::Point(xM,yM), 10, cv::Scalar(0,0,255), 2);
    //cv::rectangle(matOriginal, cv::Point(xM, yM), cv::Point(x1M, y1M), cv::Scalar(0,0,255), 2);



    ///////////////////////////////////////
    /// \brief Creation de la mouth map
    ///////////////////////////////////////

    float extent = a / (widthOfHead * heightOfHead);


    if(centX != 0 && centY != 0){
        int lowX = x1 - (x1 * 0.4);
        int highX = x + (x1 * 0.4);

        int lowY = y + (y1 - y) * 0.4;
        int highY = y + (y1 - y) * 0.6;

        bool centroidCheck = (centX < highX) && (centX > lowX) && (centY > lowY) && (centY < highY);

    //    cv::rectangle(matOriginal, cv::Point(highX, lowY), cv::Point(lowX, highY), cv::Scalar(128,64,18), 1);
    //    cv::rectangle(matOriginal, cv::Point(centX, centY), cv::Point(centX, centY), cv::Scalar(155,240,110), 3);

        if((extent > 0.45) && (extent < 0.90) && faceDetected)
            if(centroidCheck)
                cv::rectangle(matOriginal, cv::Point(x,y), cv::Point(x1,y1), cv::Scalar(255,0,0), 2);
    }

    cv::putText(matOriginal, "Original", cv::Point(10,230), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,0,255), 2.0);
    cv::putText(bw3, "Skin", cv::Point(10,230), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,0), 2.0);
    cv::putText(eyeMapFinal, "Eyes", cv::Point(10,230), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,0), 2.0);
    cv::putText(mouthMapFinal, "Mouth", cv::Point(10,230), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255,0,0), 2.0);

    cv::cvtColor(matOriginal, matOriginal, CV_BGR2RGB);

    QImage qimgOriginal((uchar*)matOriginal.data, matOriginal.cols, matOriginal.rows, matOriginal.step, QImage::Format_RGB888);
    QImage qimgProcessed((uchar*)bw3.data, bw3.cols, bw3.rows, bw3.step, QImage::Format_Indexed8);
    QImage qimgMouth((uchar*)mouthMapFinal.data, mouthMapFinal.cols, mouthMapFinal.rows, mouthMapFinal.step, QImage::Format_RGB888);
    QImage qimgEyes((uchar*)eyeMapFinal.data, eyeMapFinal.cols, eyeMapFinal.rows, eyeMapFinal.step, QImage::Format_RGB888);

    ui->original->setPixmap(QPixmap::fromImage(qimgOriginal));
    ui->processed->setPixmap(QPixmap::fromImage(qimgProcessed));
    ui->eyes->setPixmap(QPixmap::fromImage(qimgEyes));
    ui->mouth->setPixmap(QPixmap::fromImage(qimgMouth));

}

void Dialog::changeChannel()
{
    capWebcam.open(ui->deviceNumber->value());
    capWebcam.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    capWebcam.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    if(!capWebcam.isOpened())
    {
        QMessageBox::information(this, "Error accessing the webcam", "There was an error when trying to acces the webcam");
        return;
    }

}

void Dialog::changeSize()
{
    if(ui->hide->text() == "<"){
        this->resize(720, 506);
        ui->hide->setGeometry(670,10,41,481);
        ui->groupBox->hide();
        ui->groupBox_2->hide();
        ui->groupBox_3->hide();
        ui->groupBox_4->hide();
        ui->groupBox_5->hide();
        ui->hide->setText(">");
    }else{
        this->resize(1192, 506);
        ui->hide->setGeometry(1140,10,41,481);
        ui->groupBox->show();;
        ui->groupBox_2->show();
        ui->groupBox_3->show();
        ui->groupBox_4->show();
        ui->groupBox_5->show();
        ui->hide->setText("<");
    }
}

/////////////////////////////////////////////////////////////////////////////

void Dialog::on_starting_clicked()
{
    if(tmrTimer->isActive())
    {
        tmrTimer->stop();
        ui->starting->setText("Start video feed");
    }else
    {
        tmrTimer->start(refreshTime);
        ui->starting->setText("Stop video feed");
    }

}

