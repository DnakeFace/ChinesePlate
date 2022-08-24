#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include <stdio.h>

#include "net.h"
#include "types.h"

struct YoloObject {
	cv::Rect_<float> rect;
	int label;
	float score;
};

typedef struct {
	int width;
	int height;
	float scale;

	struct {
		int w;
		int h;
	} in;

	struct {
		int w;
		int h;
	} pad;
} yolov5_info_t;

class Yolov5 {
public:
	Yolov5(const char *model = NULL);
	~Yolov5();

	int detect(cv::Mat *d, VIPLObjectInfo *objs, int size);

	void post_stride(ncnn::Mat &data, ncnn::Mat &anchors, yolov5_info_t &info, int stride, std::vector<YoloObject> &objs);
	int post_process(yolov5_info_t &info, std::vector<YoloObject> &proposals, VIPLObjectInfo *objs, int size);

public:
	ncnn::Net m_net;

	std::vector<int> m_extract;

	int m_channels = 3;
	float m_width = 320;
	float m_height = 320;

	std::vector<ncnn::Mat> m_anchors;
	std::vector<int> m_stride;

	float m_threshold = 0.70f;
	float m_nms_threshold = 0.45f;

	int m_num_threads = 2;
};

#endif
