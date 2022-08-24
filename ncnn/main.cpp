
#include "Yolov5.h"
#include "landmark.h"
#include "crnn.h"

static inline bool transform2d(const float *mean, const float *landmark, int N, float *M)
{
	const float *std_points = landmark;
	const float *src_points = mean;

	float sum_x = 0, sum_y = 0;
	float sum_u = 0, sum_v = 0;
	float sum_xx_yy = 0;
	float sum_ux_vy = 0;
	float sum_vx_uy = 0;

	for (int c = 0; c < N; ++c) {
		int x_off = c * 2;
		int y_off = x_off + 1;
		sum_x += std_points[c * 2];
		sum_y += std_points[c * 2 + 1];
		sum_u += src_points[x_off];
		sum_v += src_points[y_off];
		sum_xx_yy += std_points[c * 2] * std_points[c * 2] + std_points[c * 2 + 1] * std_points[c * 2 + 1];
		sum_ux_vy += std_points[c * 2] * src_points[x_off] + std_points[c * 2 + 1] * src_points[y_off];
		sum_vx_uy += src_points[y_off] * std_points[c * 2] - src_points[x_off] * std_points[c * 2 + 1];
	}

	if (sum_xx_yy <= FLT_EPSILON)
		return false;

	float q = sum_u - sum_x * sum_ux_vy / sum_xx_yy + sum_y * sum_vx_uy / sum_xx_yy;
        float p = sum_v - sum_y * sum_ux_vy / sum_xx_yy - sum_x * sum_vx_uy / sum_xx_yy;
	float r = N - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;

	if (!(r > FLT_EPSILON || r < -FLT_EPSILON))
		return false;

	float a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;
	float b = (sum_vx_uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy;
	float c = q / r;
	float d = p / r;

	M[0] = M[4] = a;
	M[1] = -b;
	M[3] = b;
	M[2] = c;
	M[5] = d;

	return true;
}

static const float transform2d_mean_100x32[8] = {
	92, 30,
	8,  30,
	8,  2,
	92, 2,
};

static inline bool transform2d_100x32(VIPLPoint *p, float *M)
{
	float landmark[8];
	for (size_t i = 0; i < 4; ++i) {
		landmark[i*2] = p[i].x;
		landmark[i*2 + 1] = p[i].y;
	}
	return transform2d(transform2d_mean_100x32, landmark, 4, M);
}

int main(void)
{
	Yolov5 yolov5;
	landmark pfld;
	PlateCrnn crnn;

	cv::Mat m = cv::imread("./test/p1.jpg", 1);
	if (m.empty()) {
		fprintf(stderr, "cv::imread failed\n");
		return -1;
	}
	VIPLObjectInfo objs[100];
	int n = yolov5.detect(&m, objs, 100);

	for(int i=0; i<n; i++) {
		if (objs[i].label != 0)
			continue;

		VIPLPoint p4[4];
		pfld.detect(&m, &objs[i], p4);

		cv::Mat crop(cv::Size(100, 32), CV_8UC3);
		cv::Mat M(2, 3, CV_32FC1);
		transform2d_100x32(p4, (float *)M.data);
		cv::warpAffine(m, crop, M, crop.size());
		cv::imwrite("crop.jpg", crop);

		PlateText_t result;
		crnn.detect(&crop, &result);
		FILE *fp = fopen("result.txt", "a+");
		if (fp) {
			fprintf(fp, "%s\n", result.text);
			fclose(fp);
		}

		for(int k=0; k<4; k++) {
			cv::circle(m, cv::Point(p4[k].x, p4[k].y), 3, cv::Scalar(0, 0, 255), -1);
		}
	}

	for (int k=0; k<n; k++) {
		const VIPLObjectInfo& obj = objs[k];
		cv::Scalar c;
		if (obj.label == 0)
			c = cv::Scalar(0, 0, 255);
		else
			c = cv::Scalar(255, 0, 0);
		cv::rectangle(m, cv::Point(obj.x, obj.y), cv::Point(obj.x+obj.width, obj.y+obj.height), c);
	}

	cv::imwrite("result.jpg", m);

	return 0;
}
