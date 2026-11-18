#ifndef HAZE_REMOVAL_H
#define HAZE_REMOVAL_H

#include "guidedfilter.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>

typedef struct _pixel {
	int i;
	int j;
	uchar val;
	_pixel(int _i, int _j, uchar _val) :i(_i), j(_j), val((uchar)_val) {}
} Pixel;

struct BitRangeStats {
    double minValue;
    double maxValue;
    std::string minImage;
    std::string maxImage;

    BitRangeStats()
        : minValue(std::numeric_limits<double>::max()),
        maxValue(std::numeric_limits<double>::lowest()) {
    }

    void update(double localMin, double localMax,
        const std::string& imageName) {
        if (localMin < minValue) {
            minValue = localMin;
            minImage = imageName;
        }

        if (localMax > maxValue) {
            maxValue = localMax;
            maxImage = imageName;
        }
    }
};

struct DatasetBitStats {
    BitRangeStats airlightB;
    BitRangeStats airlightG;
    BitRangeStats airlightR;
    BitRangeStats bRatio;
    BitRangeStats gRatio;
    BitRangeStats rRatio;
    BitRangeStats minpixel;
    BitRangeStats minVal;
    BitRangeStats transmission;
};

class CHazeRemoval
{
public:
	CHazeRemoval();
	~CHazeRemoval();

public:
	bool InitProc(int width, int height, int nChannels);
	bool Process(const unsigned char* indata, unsigned char* outdata, int width, int height, int nChannels);

private:
	int rows;
	int cols;
	int channels;

};

void get_dark_channel(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, int rows, int cols, int channels, int radius);

void get_air_light(const cv::Mat *p_src, std::vector<Pixel> &tmp_vec, cv::Vec3d *p_Alight, int rows, int cols, int channels);

void get_transmission(const cv::Mat *p_src, cv::Mat *p_tran, cv::Vec3d *p_Alight, int rows, int cols, int channels, int radius, double omega);

void get_transmission_measurement(
    const cv::Mat* p_src,
    cv::Mat* p_tran,
    cv::Vec3d* p_Alight,
    int rows,
    int cols,
    int channels,
    int radius,
    double omega,
    DatasetBitStats* datasetStats = nullptr,
    const std::string& imageName = "",
    bool printPerImage = true
);

void guided_filter(const cv::Mat *p_src, const cv::Mat *p_tran, cv::Mat *p_gtran, int r, double eps);

void recover(const cv::Mat *p_src, const cv::Mat *p_gtran, cv::Mat *p_dst, cv::Vec3d *p_Alight, int rows, int cols, int channels, double t0);

void assign_data(unsigned char *outdata, const cv::Mat *p_dst, int rows, int cols, int channels);

#endif // !HAZE_REMOVAL_H