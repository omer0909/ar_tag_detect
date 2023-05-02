#include <memory>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "opencv_demo/msg/ar_info.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/static_transform_broadcaster.h"

class ImageProcessTest : public rclcpp::Node {
   public:
	ImageProcessTest() : Node("image_process_test") {
		auto event = std::bind(&ImageProcessTest::ArTagEvent, this, std::placeholders::_1);
		subscriberArTag = this->create_subscription<opencv_demo::msg::ArInfo>("/ar_tag_pose", 10, event);

		tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
	}

	void ArTagEvent(const opencv_demo::msg::ArInfo::SharedPtr arTag) {
		geometry_msgs::msg::TransformStamped t;

		t.header.stamp = this->get_clock()->now();
		t.header.frame_id = "camera_link";
		std::stringstream frame_id;
		frame_id << "arTag_" << arTag->id;
		t.child_frame_id = frame_id.str();

		t.transform.translation.x = arTag->pose.position.x;
		t.transform.translation.y = arTag->pose.position.y;
		t.transform.translation.z = arTag->pose.position.z;

		t.transform.rotation = arTag->pose.orientation;

		tf_static_broadcaster_->sendTransform(t);
	}

	void make_transforms() {
	}

	rclcpp::Subscription<opencv_demo::msg::ArInfo>::SharedPtr subscriberArTag;
	std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
};

int main(int argc, char* argv[]) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<ImageProcessTest>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
