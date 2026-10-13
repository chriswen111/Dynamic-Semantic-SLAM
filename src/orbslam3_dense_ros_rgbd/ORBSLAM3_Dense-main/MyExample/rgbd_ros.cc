#include <exception>
#include <memory>
#include <string>

#include <csignal> 

#include <boost/bind/bind.hpp>

#include <opencv2/core/core.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "System.h"

ORB_SLAM3::System* g_slam_system = nullptr;

// 信号处理函数
void sigint_handler(int sig)
{
    if (g_slam_system)
    {
        ROS_WARN("Ctrl+C detected, shutting down SLAM system to save map...");
        g_slam_system->Shutdown();
    }
    ros::shutdown();
}

class RGBDGrabber
{
public:
    explicit RGBDGrabber(ORB_SLAM3::System *slam) : slam_(slam), frame_count_(0) {}

    void GrabRGBD(const sensor_msgs::ImageConstPtr &rgb_msg,
                  const sensor_msgs::ImageConstPtr &depth_msg)
    {
        Process(rgb_msg, depth_msg, sensor_msgs::ImageConstPtr());
    }

    void GrabRGBDSemantic(const sensor_msgs::ImageConstPtr &rgb_msg,
                          const sensor_msgs::ImageConstPtr &depth_msg,
                          const sensor_msgs::ImageConstPtr &mask_msg)
    {
        Process(rgb_msg, depth_msg, mask_msg);
    }

private:
    void Process(const sensor_msgs::ImageConstPtr &rgb_msg,
                 const sensor_msgs::ImageConstPtr &depth_msg,
                 const sensor_msgs::ImageConstPtr &mask_msg)
    {
        ++frame_count_;
        ROS_INFO_STREAM_THROTTLE(2.0,
                                 "RGB-D synced frame #" << frame_count_
                                                        << " rgb_stamp=" << rgb_msg->header.stamp.toSec()
                                                        << " depth_stamp=" << depth_msg->header.stamp.toSec()
                                                        << (mask_msg ? " mask_stamp=" + std::to_string(mask_msg->header.stamp.toSec()) : "")
                                                        << " rgb_encoding=" << rgb_msg->encoding
                                                        << " depth_encoding=" << depth_msg->encoding
                                                        << (mask_msg ? " mask_encoding=" + mask_msg->encoding : "")
                                                        << " rgb_size=" << rgb_msg->width << "x" << rgb_msg->height
                                                        << " depth_size=" << depth_msg->width << "x" << depth_msg->height
                                                        << (mask_msg ? " mask_size=" + std::to_string(mask_msg->width) + "x" + std::to_string(mask_msg->height) : ""));

        cv_bridge::CvImageConstPtr cv_rgb;
        cv_bridge::CvImageConstPtr cv_depth;
        cv_bridge::CvImageConstPtr cv_mask;

        try
        {
            cv_rgb = cv_bridge::toCvShare(rgb_msg);
            cv_depth = cv_bridge::toCvShare(depth_msg);
            if (mask_msg)
                cv_mask = cv_bridge::toCvShare(mask_msg);
        }
        catch (const cv_bridge::Exception &e)
        {
            ROS_ERROR_STREAM_THROTTLE(1.0, "cv_bridge exception: " << e.what());
            return;
        }

        if (cv_rgb->image.empty() || cv_depth->image.empty())
        {
            ROS_WARN_THROTTLE(1.0, "Received empty RGB or depth image.");
            return;
        }

        if (cv_rgb->image.channels() != 1 && cv_rgb->image.channels() != 3 && cv_rgb->image.channels() != 4)
        {
            ROS_WARN_STREAM_THROTTLE(1.0, "Unsupported RGB image channels: " << cv_rgb->image.channels());
            return;
        }

        cv::Mat semantic_mask;
        if (cv_mask)
        {
            if (cv_mask->image.empty())
            {
                ROS_WARN_THROTTLE(1.0, "Received empty semantic mask image.");
                return;
            }

            if (cv_mask->image.type() != CV_16UC1)
            {
                ROS_WARN_STREAM_THROTTLE(1.0,
                                         "Semantic mask must be mono16/16UC1, got encoding "
                                             << mask_msg->encoding << " and OpenCV type " << cv_mask->image.type());
                return;
            }

            if (cv_mask->image.size() != cv_rgb->image.size())
            {
                ROS_WARN_STREAM_THROTTLE(1.0,
                                         "Semantic mask size " << cv_mask->image.cols << "x" << cv_mask->image.rows
                                                              << " does not match RGB size " << cv_rgb->image.cols
                                                              << "x" << cv_rgb->image.rows);
                return;
            }

            semantic_mask = cv_mask->image;
        }

        const std::string &depth_encoding = depth_msg->encoding;
        if (depth_encoding != sensor_msgs::image_encodings::TYPE_16UC1 &&
            depth_encoding != sensor_msgs::image_encodings::TYPE_32FC1 &&
            depth_encoding != sensor_msgs::image_encodings::MONO16)
        {
            ROS_WARN_STREAM_THROTTLE(2.0,
                                     "Depth encoding is " << depth_encoding
                                                          << ". Expected 16UC1, 32FC1, or mono16.");
        }

        const ros::Time stamp = rgb_msg->header.stamp.isZero() ? depth_msg->header.stamp : rgb_msg->header.stamp;
        const double timestamp = stamp.isZero() ? ros::Time::now().toSec() : stamp.toSec();

        try
        {
            if (semantic_mask.empty())
                slam_->TrackRGBD(cv_rgb->image, cv_depth->image, timestamp);
            else
                slam_->TrackRGBD(cv_rgb->image, cv_depth->image, semantic_mask, timestamp);
        }
        catch (const cv::Exception &e)
        {
            ROS_ERROR_STREAM_THROTTLE(1.0, "OpenCV exception while tracking: " << e.what());
        }
        catch (const std::exception &e)
        {
            ROS_ERROR_STREAM_THROTTLE(1.0, "Exception while tracking: " << e.what());
        }
    }

    ORB_SLAM3::System *slam_;
    size_t frame_count_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "orbslam3_rgbd");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    const std::string package_path = ros::package::getPath("orbslam3_dense");
    const std::string default_voc = package_path.empty() ? "Vocabulary/ORBvoc.txt" : package_path + "/Vocabulary/ORBvoc.txt";
    const std::string default_settings = package_path.empty() ? "MyExample/rgbd_orb.yaml" : package_path + "/MyExample/rgbd_orb.yaml";

    std::string voc_file;
    std::string settings_file;
    std::string rgb_topic;
    std::string depth_topic;
    std::string semantic_mask_topic;
    bool enable_semantic_mask;
    bool use_viewer;
    int queue_size;
    double sync_slop;

    private_nh.param<std::string>("voc_file", voc_file, default_voc);
    private_nh.param<std::string>("settings_file", settings_file, default_settings);
    private_nh.param<std::string>("rgb_topic", rgb_topic, "/camera/color/image_raw");
    private_nh.param<std::string>("depth_topic", depth_topic, "/camera/depth/image_raw");
    private_nh.param<std::string>("semantic_mask_topic", semantic_mask_topic, "/semantic_mask");
    private_nh.param<bool>("enable_semantic_mask", enable_semantic_mask, true);
    private_nh.param<bool>("use_viewer", use_viewer, true);
    private_nh.param<int>("queue_size", queue_size, 10);
    private_nh.param<double>("sync_slop", sync_slop, 0.03);

    ROS_INFO_STREAM("Vocabulary: " << voc_file);
    ROS_INFO_STREAM("Settings: " << settings_file);
    ROS_INFO_STREAM("RGB topic: " << rgb_topic);
    ROS_INFO_STREAM("Depth topic: " << depth_topic);
    ROS_INFO_STREAM("Semantic mask: " << (enable_semantic_mask ? semantic_mask_topic : "disabled"));

    ORB_SLAM3::System slam(voc_file, settings_file, ORB_SLAM3::System::RGBD, use_viewer);

    // 将全局指针指向当前 slam 实例
    g_slam_system = &slam;
    
    // 注册 SIGINT 信号处理函数 (Ctrl+C)
    signal(SIGINT, sigint_handler);


    RGBDGrabber grabber(&slam);

    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter rgb_sub(it, rgb_topic, queue_size);
    image_transport::SubscriberFilter depth_sub(it, depth_topic, queue_size);
    std::shared_ptr<image_transport::SubscriberFilter> mask_sub;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> RGBDSyncPolicy;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> SemanticSyncPolicy;

    std::shared_ptr<message_filters::Synchronizer<RGBDSyncPolicy> > rgbd_sync;
    std::shared_ptr<message_filters::Synchronizer<SemanticSyncPolicy> > semantic_sync;

    if (enable_semantic_mask)
    {
        mask_sub.reset(new image_transport::SubscriberFilter(it, semantic_mask_topic, queue_size));
        semantic_sync.reset(new message_filters::Synchronizer<SemanticSyncPolicy>(SemanticSyncPolicy(queue_size), rgb_sub, depth_sub, *mask_sub));
        semantic_sync->setMaxIntervalDuration(ros::Duration(sync_slop));
        semantic_sync->registerCallback(boost::bind(&RGBDGrabber::GrabRGBDSemantic,
                                                    &grabber,
                                                    boost::placeholders::_1,
                                                    boost::placeholders::_2,
                                                    boost::placeholders::_3));
    }
    else
    {
        rgbd_sync.reset(new message_filters::Synchronizer<RGBDSyncPolicy>(RGBDSyncPolicy(queue_size), rgb_sub, depth_sub));
        rgbd_sync->setMaxIntervalDuration(ros::Duration(sync_slop));
        rgbd_sync->registerCallback(boost::bind(&RGBDGrabber::GrabRGBD,
                                                &grabber,
                                                boost::placeholders::_1,
                                                boost::placeholders::_2));
    }

    ros::spin();

    if (ros::ok()) {
        slam.Shutdown();
    }
    return 0;
}
