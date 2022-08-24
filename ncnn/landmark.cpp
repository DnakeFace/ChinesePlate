
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "landmark.h"

landmark::landmark(const char *model)
{
	char s1[256], s2[256];
	if (model) {
		sprintf(s1, "./model/%s.param", model);
		sprintf(s2, "./model/%s.bin", model);
	} else {
		sprintf(s1, "./model/landmark.param");
		sprintf(s2, "./model/landmark.bin");
	}
	m_net.load_param(s1);
	m_net.load_model(s2);
	m_extract = m_net.output_indexes();
}

landmark::~landmark()
{
}

int landmark::detect(cv::Mat *d, VIPLObjectInfo *f, VIPLPoint *p, float crop)
{
	ncnn::Extractor ex = m_net.create_extractor();
	ex.set_num_threads(m_num_threads);

	int cx = f->x + f->width/2;
	int cy = f->y + f->height/2;
	int mwh = crop * std::max(f->width, f->height);
	int x = std::max(0, cx - mwh/2);
	int y = std::max(0, cy - mwh/2);
	int x2 = std::min(cx + mwh/2, d->cols);
	int y2 = std::min(cy + mwh/2, d->rows);

	ncnn::Mat border, in;
	ncnn::Mat roi = ncnn::Mat::from_pixels_roi(d->data, ncnn::Mat::PIXEL_BGR, d->cols, d->rows, x, y, x2-x, y2-y);

	mwh = std::max(roi.w, roi.h);
	int sx1 = std::abs((cx - x) - mwh/2);
	int sx2 = std::abs((cx - x) + mwh/2 - roi.w);
	int sy1 = std::abs((cy - y) - mwh/2);
	int sy2 = std::abs((cy - y) + mwh/2 - roi.h);
	ncnn::copy_make_border(roi, border, sy1, sy2, sx1, sx2, ncnn::BORDER_CONSTANT, 0);

	#define DST_SIZE	(112)
	ncnn::resize_bilinear(border, in, DST_SIZE, DST_SIZE);

	const float mean[3] = { 0.0, 0.0, 0.0 };
	const float norm[3] = { 1/255.0, 1/255.0, 1/255.0 };
	in.substract_mean_normalize(mean, norm);
	ex.input(0, in);

	ncnn::Mat out;
	ex.extract(m_extract[0], out);
	float *lm = out.channel(0);
	for(int i=0; i<(out.w/2); i++) {
		p[i].x = lm[2*i]*border.w - sx1 + x;
		p[i].y = lm[2*i+1]*border.w - sy1 + y;
	}
	return 0;
}
