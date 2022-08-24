
#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>
#include <float.h>

#define AW_API

#ifndef PI
#define PI (3.1415926)
#endif

#include "opencv2/opencv.hpp"

typedef struct VIPLImageData {
	VIPLImageData()
	{
		data = NULL;
		width = 0;
		height = 0;
		channels = 0;
	}

	VIPLImageData(int32_t _width, int32_t _height, int32_t _channels = 1)
	{
		data = NULL;
		width = _width;
		height = _height;
		channels = _channels;
	}

	uint8_t* data;
	int32_t width;
	int32_t height;
	int32_t channels;
} VIPLImageData;

typedef struct {
	int x;
	int y;
	int width;
	int height;
	int label;
	float score;
} VIPLObjectInfo;

typedef struct {
	double x;
	double y;
} VIPLPoint;

typedef struct {
	int size;
	char text[128];
	float score[26];
} PlateText_t;

static inline long __ts(struct timeval &tv)
{
	struct timeval tv2;
	gettimeofday(&tv2, NULL);
	return labs((tv2.tv_sec-tv.tv_sec)*1000+(tv2.tv_usec-tv.tv_usec)/1000);
}

static inline void __swap(int &w, int &h)
{
	int t = w;
	w = h;
	h = t;
}

static inline int ALIGN(int x, int y)
{
	return (x + (y-1)) & (~(y-1));
}

#endif
