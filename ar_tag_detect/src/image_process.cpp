#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <image_transport/image_transport.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "ar_tag_detect_interfaces/msg/ar_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

#define ANGLE_TO_RADIAN (M_PI / 180)

std::map<std::string, int> ARUCO_DICT{
    {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
    {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
    {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
    {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
    {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
    {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
    {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
    {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
    {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
    {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
    {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
    {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
    {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
    {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
    {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
    {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
    {"DICT_APRILTAG_16h5", cv::aruco::DICT_APRILTAG_16h5},
    {"DICT_APRILTAG_25h9", cv::aruco::DICT_APRILTAG_25h9},
    {"DICT_APRILTAG_36h10", cv::aruco::DICT_APRILTAG_36h10},
    {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11}};

std::string aruco_type = "DICT_5X5_100";
cv::Ptr<cv::aruco::Dictionary> arucoDict = cv::aruco::getPredefinedDictionary(ARUCO_DICT[aruco_type]);

using namespace std::chrono_literals;

class ImageProcess : public rclcpp::Node {
   public:
	ImageProcess() : Node("image_process") {
		auto imageEvent = std::bind(&ImageProcess::ImageUpdate, this, std::placeholders::_1);
		cameraView_ = this->create_subscription<sensor_msgs::msg::Image>("/camera1/image_raw", 10, imageEvent);

		auto cameraInfoEvent = std::bind(&ImageProcess::CameraInfoUpdate, this, std::placeholders::_1);
		cameraInfo_ = this->create_subscription<sensor_msgs::msg::CameraInfo>("/camera1/camera_info", 10, cameraInfoEvent);

		publisherArTagPose_ = this->create_publisher<ar_tag_detect_interfaces::msg::ArInfo>("/ar_tag_pose", 1);
	}

   private:
	void cameraInfoToCV(const sensor_msgs::msg::CameraInfo::SharedPtr& msg,
	                    cv::Matx33d& K_,       // Describe current image (includes binning, ROI)
	                    cv::Mat_<double>& D_)  // Unaffected by binning, ROI - they are in ideal camera coordinates
	{
		// TODO(lucasw) this can't be const
		auto cam_info = *msg;

		cv::Matx34d P_;  // Describe current image (includes binning, ROI)

		int d_size = cam_info.d.size();
		double arr[5];
		std::copy(cam_info.d.begin(), cam_info.d.end(), arr);
		D_ = (d_size == 0) ? cv::Mat_<double>() : cv::Mat_<double>(1, d_size, arr);

		// std::stringstream sstr;
		// sstr << D_;
		// RCLCPP_INFO(this->get_logger(), "uuuu \n %s \n", sstr.str().c_str());

		auto K_full_ = cv::Matx33d(&cam_info.k[0]);
		// TODO(lucasw) not actually using P_full_
		auto P_full_ = cv::Matx34d(&cam_info.p[0]);

		// Binning = 0 is considered the same as binning = 1 (no binning).
		const uint32_t binning_x = cam_info.binning_x ? cam_info.binning_x : 1;
		const uint32_t binning_y = cam_info.binning_y ? cam_info.binning_y : 1;

		// ROI all zeros is considered the same as full resolution.
		sensor_msgs::msg::RegionOfInterest roi = cam_info.roi;
		if (roi.x_offset == 0 && roi.y_offset == 0 && roi.width == 0 && roi.height == 0) {
			roi.width = cam_info.width;
			roi.height = cam_info.height;
		}

		// If necessary, create new K_ and P_ adjusted for binning and ROI
		/// @todo Calculate and use rectified ROI
		const bool adjust_binning = (binning_x > 1) || (binning_y > 1);
		const bool adjust_roi = (roi.x_offset != 0) || (roi.y_offset != 0);

		if (!adjust_binning && !adjust_roi) {
			K_ = K_full_;
			P_ = P_full_;
		} else {
			K_ = K_full_;
			P_ = P_full_;

			// ROI is in full image coordinates, so change it first
			if (adjust_roi) {
				// Move principal point by the offset
				/// @todo Adjust P by rectified ROI instead
				K_(0, 2) -= roi.x_offset;
				K_(1, 2) -= roi.y_offset;
				P_(0, 2) -= roi.x_offset;
				P_(1, 2) -= roi.y_offset;
			}

			if (binning_x > 1) {
				const double scale_x = 1.0 / binning_x;
				K_(0, 0) *= scale_x;
				K_(0, 2) *= scale_x;
				P_(0, 0) *= scale_x;
				P_(0, 2) *= scale_x;
				P_(0, 3) *= scale_x;
			}
			if (binning_y > 1) {
				const double scale_y = 1.0 / binning_y;
				K_(1, 1) *= scale_y;
				K_(1, 2) *= scale_y;
				P_(1, 1) *= scale_y;
				P_(1, 2) *= scale_y;
				P_(1, 3) *= scale_y;
			}
		}
	}

	void ImageUpdate(const sensor_msgs::msg::Image::SharedPtr imageMsg) {
		if (!cameraInfoTaked)
			return;

		cv::Mat img;
		try {
			cv::Mat img = cv_bridge::toCvShare(imageMsg, "bgr8")->image;

			std::vector<int> ids;
			std::vector<std::vector<cv::Point2f>> corners;

			cv::aruco::detectMarkers(img, arucoDict, corners, ids);

			// if (ids.size() > 0)
			// 	cv::aruco::drawDetectedMarkers(img, corners, ids);

			int nMarkers = corners.size();
			std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

			float markerLength = 0.035;

			cv::Mat objPoints(4, 1, CV_32FC3);
			objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength / 2.f, markerLength / 2.f, 0);
			objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerLength / 2.f, markerLength / 2.f, 0);
			objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerLength / 2.f, -markerLength / 2.f, 0);
			objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength / 2.f, -markerLength / 2.f, 0);

			// Calculate pose for each marker
			for (int i = 0; i < nMarkers; i++) {
				cv::solvePnP(objPoints, corners.at(i), cameraMatrix_, distCoeffs_, rvecs.at(i), tvecs.at(i));
			}

			// Draw axis for each marker
			for (unsigned int i = 0; i < ids.size(); i++) {
				ar_tag_detect_interfaces::msg::ArInfo arTagPose;

				// std::stringstream out;
				// out << "rvecs: " << rvecs[i] << std::endl
				//     << "tvecs: " << tvecs[i] << std::endl;
				// RCLCPP_INFO(this->get_logger(), out.str().c_str());

				// cv::Quatd rotate = cv::Quatd::createFromRvec(cv::Vec3d(0, 90 * ANGLE_TO_RADIAN, 90 * ANGLE_TO_RADIAN));

				cv::Quatd rotate(0.5, -0.5, 0.5, -0.5);

				cv::Quatd q = cv::Quatd::createFromRvec(rvecs[i]);
				q = rotate * q;

				cv::Quatd p = cv::Quatd(0, tvecs[i][0], tvecs[i][1], tvecs[i][2]);
				p = (rotate * p) * rotate.conjugate();

				arTagPose.id = ids[i];

				arTagPose.pose.orientation.x = q.x;
				arTagPose.pose.orientation.y = q.y;
				arTagPose.pose.orientation.z = q.z;
				arTagPose.pose.orientation.w = q.w;

				arTagPose.pose.position.x = p.x;
				arTagPose.pose.position.y = p.y;
				arTagPose.pose.position.z = p.z;

				publisherArTagPose_->publish(arTagPose);
				// cv::drawFrameAxes(img, cameraMatrix_, distCoeffs_, rvecs[i], tvecs[i], 0.1);
			}

			// std::stringstream sstr;
			// sstr << cameraMatrix_;
			// sstr << "\n--------------------------------------\n";
			// sstr << distCoeffs_;
			// RCLCPP_INFO(this->get_logger(), "resim_geldi \n %s \n", sstr.str().c_str());

			// cv::imshow("Image Viewer", img);
			// cv::waitKey(1);
		} catch (cv_bridge::Exception& e) {
			RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
			return;
		}
		// RCLCPP_INFO(this->get_logger(), "resim_geldi");
	}

	void CameraInfoUpdate(const sensor_msgs::msg::CameraInfo::SharedPtr cameraInfoMsg) {
		cameraInfoToCV(cameraInfoMsg, cameraMatrix_, distCoeffs_);

		// cameraMatrix_ = cv::Mat(3, 3, CV_64FC1, (void*)cameraInfoMsg->k.data());
		// RCLCPP_INFO(this->get_logger(), "camera_info_geldi");
		cameraInfoTaked = true;
	}

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr cameraView_;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cameraInfo_;
	cv::Matx33d cameraMatrix_;
	cv::Mat_<double> distCoeffs_;
	bool cameraInfoTaked = false;
	rclcpp::Publisher<ar_tag_detect_interfaces::msg::ArInfo>::SharedPtr publisherArTagPose_;
};

int main(int argc, char* argv[]) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<ImageProcess>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
