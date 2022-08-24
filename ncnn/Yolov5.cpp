// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <float.h>
#include <stdio.h>
#include <vector>
#include <fstream>

#include "Yolov5.h"

static inline float intersection_area(const YoloObject& a, const YoloObject& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(std::vector<YoloObject>& faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].score;

	while (i <= j) {
		while (faceobjects[i].score > p)
			i++;

		while (faceobjects[j].score < p)
			j--;

		if (i <= j) {
			// swap
			std::swap(faceobjects[i], faceobjects[j]);
			i++;
			j--;
		}
	}

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			if (left < j) qsort_descent_inplace(faceobjects, left, j);
		}
		#pragma omp section
		{
			if (i < right) qsort_descent_inplace(faceobjects, i, right);
		}
	}
}

static void qsort_descent_inplace(std::vector<YoloObject>& faceobjects)
{
	if (faceobjects.empty())
		return;
	qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<YoloObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++) {
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++) {
		const YoloObject& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++) {
			const YoloObject& b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int width, int height, int stride, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<YoloObject>& objects)
{
	const int num_grid = feat_blob.h;

	int num_grid_x;
	int num_grid_y;
	if (width > height) {
		num_grid_x = width / stride;
		num_grid_y = num_grid / num_grid_x;
	} else {
		num_grid_y = height / stride;
		num_grid_x = num_grid / num_grid_y;
	}

	const int num_class = feat_blob.w - 5;
	const int num_anchors = anchors.w / 2;
	for (int q = 0; q < num_anchors; q++) {
		const float anchor_w = anchors[q * 2];
		const float anchor_h = anchors[q * 2 + 1];

		const ncnn::Mat feat = feat_blob.channel(q);

		for (int i = 0; i < num_grid_y; i++) {
			for (int j = 0; j < num_grid_x; j++) {
				const float* featptr = feat.row(i * num_grid_x + j);

				// find class index with max class score
				int class_index = 0;
				float class_score = -FLT_MAX;
				for (int k = 0; k < num_class; k++) {
					float score = featptr[5 + k];
					if (score > class_score) {
						class_index = k;
						class_score = score;
					}
				}

				float box_score = featptr[4];

				float confidence = sigmoid(box_score) * sigmoid(class_score);

				if (confidence >= prob_threshold) {
					// yolov5/models/yolo.py Detect forward
					// y = x[i].sigmoid()
					// y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
					// y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

					float dx = sigmoid(featptr[0]);
					float dy = sigmoid(featptr[1]);
					float dw = sigmoid(featptr[2]);
					float dh = sigmoid(featptr[3]);

					float pb_cx = (dx * 2.f - 0.5f + j) * stride;
					float pb_cy = (dy * 2.f - 0.5f + i) * stride;

					float pb_w = pow(dw * 2.f, 2) * anchor_w;
					float pb_h = pow(dh * 2.f, 2) * anchor_h;

					float x0 = pb_cx - pb_w * 0.5f;
					float y0 = pb_cy - pb_h * 0.5f;
					float x1 = pb_cx + pb_w * 0.5f;
					float y1 = pb_cy + pb_h * 0.5f;

					YoloObject obj;
					obj.rect.x = x0;
					obj.rect.y = y0;
					obj.rect.width = x1 - x0;
					obj.rect.height = y1 - y0;
					obj.label = class_index;
					obj.score = confidence;

					objects.push_back(obj);
				}
			}
		}
	}
}

static inline std::string abs_path(const char *url)
{
	char s[PATH_MAX] = "";
	::realpath(url, s);
	char *last = ::strrchr(s, '/');
	if (last) {
		last[1] = 0;
	}
	return s;
}

static inline std::string abs_model(std::string root, std::string dm)
{
	std::string ss = "@file@";
	size_t n = dm.find(ss);
	if (n != std::string::npos) {
		dm = dm.replace(n, ss.length(), root);
	} else {
		dm = root + dm;
	}
	return dm;
}

Yolov5::Yolov5(const char *model)
{
	char s1[256], s2[256];
	if (model) {
		sprintf(s1, "./model/%s.param", model);
		sprintf(s2, "./model/%s.bin", model);
	} else {
		sprintf(s1, "./model/yolov5s.param");
		sprintf(s2, "./model/yolov5s.bin");
	}
	m_net.load_param(s1);
	m_net.load_model(s2);
	m_extract = m_net.output_indexes();

	ncnn::Mat anchors(6);
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;
	m_anchors.push_back(anchors.clone());
	m_stride.push_back(8);

	anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;
	m_anchors.push_back(anchors.clone());
	m_stride.push_back(16);

	anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;
	m_anchors.push_back(anchors.clone());
	m_stride.push_back(32);
}

Yolov5::~Yolov5()
{
}

int Yolov5::detect(cv::Mat *d, VIPLObjectInfo *objs, int size)
{
	yolov5_info_t info;
	info.width = d->cols;
	info.height = d->rows;

	// letterbox pad to multiple of 32
	info.scale = std::min(m_width/info.width, m_height/info.height);
	info.scale = std::min(info.scale, 1.0f);
	int dw = info.scale * info.width;
	int dh = info.scale * info.height;

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(d->data, ncnn::Mat::PIXEL_BGR2RGB, info.width, info.height, dw, dh);
	ncnn::Mat in_pad;


	info.pad.w = (dw + 31) / 32 * 32 - dw;
	info.pad.h = (dh + 31) / 32 * 32 - dh;

	ncnn::copy_make_border(in, in_pad, info.pad.h / 2, info.pad.h - info.pad.h / 2, info.pad.w / 2, info.pad.w - info.pad.w / 2, ncnn::BORDER_CONSTANT, 114.f);
	info.in.w = in_pad.w;
	info.in.h = in_pad.h;

	std::vector<YoloObject> proposals;

	const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_num_threads(m_num_threads);
	ex.input(0, in_pad);

	for (int k=0; k<m_extract.size(); k++) {
		ncnn::Mat out;
		ex.extract(m_extract[k], out);
		this->post_stride(out, m_anchors[k], info, m_stride[k], proposals);
	}

	return this->post_process(info, proposals, objs, size);
}

void Yolov5::post_stride(ncnn::Mat &data, ncnn::Mat &anchors, yolov5_info_t &info, int stride, std::vector<YoloObject> &objs)
{
	std::vector<YoloObject> objects;
	generate_proposals(anchors, info.in.w, info.in.h, stride, data, m_threshold, objects);
	objs.insert(objs.end(), objects.begin(), objects.end());
}

int Yolov5::post_process(yolov5_info_t &info, std::vector<YoloObject> &proposals, VIPLObjectInfo *objs, int size)
{
	int width = info.width;
	int height = info.height;
	int wpad = info.pad.w;
	int hpad = info.pad.h;
	float scale = info.scale;

	// sort all proposals by score from highest to lowest
	qsort_descent_inplace(proposals);

	// apply nms with nms_threshold
	std::vector<int> picked;
	nms_sorted_bboxes(proposals, picked, m_nms_threshold);

	int count = picked.size();

	std::vector<YoloObject> objects;
	objects.resize(count);
	for (int i=0; i<count; i++) {
		objects[i] = proposals[picked[i]];

		// adjust offset to original unpadded
		float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
		float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

		// clip
		x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
	count = std::min(count, size);
	for (int i=0; i<count; i++) {
		objs[i].x = objects[i].rect.x;
		objs[i].y = objects[i].rect.y;
		objs[i].width = objects[i].rect.width;
		objs[i].height = objects[i].rect.height;
		objs[i].label = objects[i].label;
		objs[i].score = objects[i].score;
	}
	return count;
}
