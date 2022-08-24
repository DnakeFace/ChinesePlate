#ifndef __POSE_ESTIMATION_H__
#define __POSE_ESTIMATION_H__

#include <sys/time.h>

#include "net.h"
#include "types.h"

class landmark {
public:
	landmark(const char *model = NULL);
	~landmark();

	int detect(cv::Mat *d, VIPLObjectInfo *f, VIPLPoint *p, float crop = 1.4);

public:
	ncnn::Net m_net;

	std::vector<int> m_extract;

	int m_num_threads = 2;
};

#endif
