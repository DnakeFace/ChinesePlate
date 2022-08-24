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

#include "crnn.h"

static const char *plate_text[] = {
" ",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"0",
"A",
"B",
"C",
"D",
"E",
"F",
"G",
"H",
"I",
"J",
"K",
"L",
"M",
"N",
"O",
"P",
"Q",
"R",
"S",
"T",
"U",
"V",
"W",
"X",
"Y",
"Z",
"京",
"津",
"沪",
"渝",
"冀",
"豫",
"云",
"辽",
"黑",
"湘",
"皖",
"鲁",
"新",
"苏",
"浙",
"赣",
"桂",
"甘",
"晋",
"蒙",
"陕",
"吉",
"闽",
"贵",
"粤",
"青",
"藏",
"琼",
"宁",
"川",
"鄂",
"港",
"澳",
"使",
"领",
"学",
"警",
"挂",
};

static void label2text(ncnn::Mat &data, PlateText_t *pt)
{
	pt->size = 0;
	memset(pt->text, 0, sizeof(pt->text));

	std::vector<int> label;
	std::vector<float> score;
	for (int k=0; k<data.h; k++) {
		float d = 0;
		int idx = -1;
		for(int i=0; i<data.w; i++) {
			if (d < data.row(k)[i]) {
				d = data.row(k)[i];
				idx = i;
			}
		}
		if (idx != -1) {
			label.push_back(idx);
			score.push_back(d);
		}
	}

	std::string text = "";
	for(int i=0; i<label.size(); i++) {
		if (label[i] != 0 && !(i > 0 && label[i-1] == label[i])) {
			if (plate_text[label[i] - 1][0] != ' ') {
				text += plate_text[label[i] - 1];
				pt->score[pt->size] = score[i];
				pt->size++;
			}
		}
	}
	strcpy(pt->text, text.c_str());
}


PlateCrnn::PlateCrnn()
{
	m_net.load_param("./model/crnn.param");
	m_net.load_model("./model/crnn.bin");
}

PlateCrnn::~PlateCrnn()
{
}

int PlateCrnn::detect(cv::Mat *d, PlateText_t *pt)
{
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(d->data, ncnn::Mat::PIXEL_BGR, d->cols, d->rows, 100, 32);

	const float mean[3] = { 0, 0, 0 };
	const float norm[3] = { 1/255.f, 1/255.f, 1/255.f };
	in.substract_mean_normalize(mean, norm);

	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_num_threads(m_num_threads);

	ex.input(0, in);

	ncnn::Mat out;
	ex.extract("output", out);

	label2text(out, pt);
	return 0;
}
