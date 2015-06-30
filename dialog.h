#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QTimer>
#include <QImage>


/////////////////////////////////////////////////////////////////////////////
namespace Ui {
class Dialog;
}
/////////////////////////////////////////////////////////////////////////////

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

private slots:
    void on_starting_clicked();

public slots:
    void processFrameAndUpdateGUI();
    void changeChannel();
    void changeSize();

private:
    Ui::Dialog *ui;

    cv::VideoCapture capWebcam;
    cv::Mat3b matOriginal;
    cv::Mat3b matProcessed;

    QImage qimgOriginal;
    QImage qimgProcessed;
    QImage qimgMouth;
    QImage qimgEyes;

    std::vector<cv::Vec3f> vecCircles;
    std::vector<cv::Vec3f>::iterator itrCircles;

    QTimer* tmrTimer;

    int refreshTime;
};

#endif // DIALOG_H
