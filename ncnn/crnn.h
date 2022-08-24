#ifndef __CRNN_H__
#define __CRNN_H__

#include <stdio.h>

#include "net.h"
#include "types.h"

class PlateCrnn {
public:
	PlateCrnn();
	~PlateCrnn();

	int detect(cv::Mat *d, PlateText_t *pt);

public:
	ncnn::Net m_net;
	int m_num_threads = 2;
};

#endif
