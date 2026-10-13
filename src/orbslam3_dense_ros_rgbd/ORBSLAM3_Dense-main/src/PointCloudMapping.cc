#include "PointCloudMapping.h"
#include <pcl/common/projection_matrix.h>
// #include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <octomap/ColorOcTree.h>
#include <opencv2/highgui/highgui.hpp>
#include "Converter.h"
#include <KeyFrame.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>

namespace ORB_SLAM3
{

    void colorBySemanticClass(const uint16_t pixelValue, uint8_t &r, uint8_t &g, uint8_t &b)
    {
        static const uint8_t palette[][3] = {
            {255, 56, 56},
            {56, 255, 56},
            {56, 120, 255},
            {255, 180, 56},
            {180, 56, 255},
            {56, 255, 220},
            {255, 56, 180},
            {180, 255, 56},
            {56, 180, 255},
            {255, 255, 56},
        };

        const int classId = static_cast<int>(pixelValue / 100);
        const int idx = classId % (sizeof(palette) / sizeof(palette[0]));
        r = palette[idx][0];
        g = palette[idx][1];
        b = palette[idx][2];
    }

    int countSemanticClassPixels(const cv::Mat &semanticMask, const int classId)
    {
        if (semanticMask.empty() || semanticMask.type() != CV_16UC1)
            return 0;

        int count = 0;
        for (int y = 0; y < semanticMask.rows; ++y)
        {
            const uint16_t *row = semanticMask.ptr<uint16_t>(y);
            for (int x = 0; x < semanticMask.cols; ++x)
            {
                const uint16_t pixelValue = row[x];
                if (pixelValue > 0 && static_cast<int>(pixelValue / 100) == classId)
                    ++count;
            }
        }
        return count;
    }

    size_t removeSemanticClassPointsFromCloud(PointCloudMapping::PointCloud::Ptr cloud,
                                              KeyFrame *kf,
                                              cv::Mat &depth,
                                              cv::Mat &semanticMask,
                                              const int classId,
                                              const double unitScale)
    {
        if (cloud == nullptr || cloud->empty() || semanticMask.empty() || semanticMask.type() != CV_16UC1 ||
            semanticMask.rows != depth.rows || semanticMask.cols != depth.cols)
            return 0;

        cv::Mat depthFloat = depth;
        if (depthFloat.type() != CV_32F)
            depth.convertTo(depthFloat, CV_32F);

        const Eigen::Matrix4f Tcw = kf->GetPose().matrix();
        PointCloudMapping::PointCloud::Ptr filtered(new PointCloudMapping::PointCloud());
        filtered->points.reserve(cloud->points.size());

        size_t removed = 0;
        for (const PointCloudMapping::PointT &pt : cloud->points)
        {
            const Eigen::Vector4f Pw(pt.x, pt.y, pt.z, 1.0f);
            const Eigen::Vector4f Pc = Tcw * Pw;
            const float z = Pc[2];

            bool remove = false;
            if (z > 0.01f)
            {
                const float uFloat = kf->fx * Pc[0] / z + kf->cx;
                const float vFloat = kf->fy * Pc[1] / z + kf->cy;
                const int u = cvRound(uFloat);
                const int v = cvRound(vFloat);

                if (u >= 0 && u < semanticMask.cols && v >= 0 && v < semanticMask.rows)
                {
                    const uint16_t pixelValue = semanticMask.ptr<uint16_t>(v)[u];
                    if (pixelValue > 0 && static_cast<int>(pixelValue / 100) == classId)
                    {
                        const float measuredDepth = depthFloat.ptr<float>(v)[u] / static_cast<float>(unitScale);
                        const float depthTolerance = std::max(0.08f, measuredDepth * 0.08f);
                        if (measuredDepth > 0.01f && measuredDepth < 10.0f &&
                            std::abs(z - measuredDepth) <= depthTolerance)
                        {
                            remove = true;
                        }
                    }
                }
            }

            if (remove)
                ++removed;
            else
                filtered->points.push_back(pt);
        }

        if (removed > 0)
        {
            filtered->width = static_cast<uint32_t>(filtered->points.size());
            filtered->height = 1;
            filtered->is_dense = cloud->is_dense;
            cloud->swap(*filtered);
        }

        return removed;
    }

    pcl::PointXYZ pcl_transform(const Eigen::Matrix4f &pose, const Eigen::Vector3f &p)
    {
        pcl::PointXYZ pcl_p;
        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.matrix() = pose;
        Eigen::Vector3f e_p = transform * p;
        return pcl::PointXYZ(e_p(0), e_p(1), e_p(2));
    }

    void PointCloudMapping::addCoordinateSystem(std::shared_ptr<pcl::visualization::PCLVisualizer> viewer, const Eigen::Matrix4f &pose, const std::string &prefix)
    {

        Eigen::Vector3f o(0, 0, 0);
        Eigen::Vector3f x_axis(this->unit * 0.25, 0, 0);
        Eigen::Vector3f y_axis(0, this->unit * 0.25, 0);
        Eigen::Vector3f z_axis(0, 0, this->unit * 0.25);

        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(pcl_transform(pose, o), pcl_transform(pose, x_axis), 1.0, 0.0, 0.0, prefix + "_x");
        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(pcl_transform(pose, o), pcl_transform(pose, y_axis), 0.0, 1.0, 0.0, prefix + "_y");
        viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(pcl_transform(pose, o), pcl_transform(pose, z_axis), 0.0, 0.0, 1.0, prefix + "_z");
    }

    PointCloudMapping::PointCloudMapping(double resolution_, double meank_, double stdthresh_, double unit_, double octomapResolution_, bool removeClass0InDenseMap_)
    {
        std::cout << "initializa with disp images !" << std::endl;
        std::cout << "the resolution of Point cloud Voxel filter : " << resolution_ << std::endl;
        this->resolution = resolution_;
        this->octomapResolution = octomapResolution_ > 0.0 ? octomapResolution_ : resolution_;
        this->removeClass0InDenseMap = removeClass0InDenseMap_;
        voxel.setLeafSize(resolution, resolution, resolution);
        std::cout << "the Point cloud Outlier filter params :   \n"
                  << "meank : " << meank_ << std::endl
                  << "stdthresh :" << stdthresh_ << std::endl;
        this->meank = meank_;
        this->stdthresh = stdthresh_;
        sor.setMeanK(meank);
        sor.setStddevMulThresh(stdthresh);
        std::cout << "the Unit of Point cloud : " << unit_ << std::endl;
        this->unit = unit_;
        std::cout << "the resolution of Semantic OctoMap : " << this->octomapResolution << std::endl;
        std::cout << "remove class-0 points in dense semantic map : " << this->removeClass0InDenseMap << std::endl;
        globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        semanticGlobalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        viewerThread = std::make_unique<std::thread>(std::bind(&PointCloudMapping::viewer, this));
    }

    PointCloudMapping::PointCloudMapping(double resolution_, double meank_, double stdthresh_, double unit_, double mindisp_, double maxdisp_, Stereo_Algorithm::AlgorithmType type, double octomapResolution_, bool removeClass0InDenseMap_)
    {
        std::cout << "initializa without disp images ,so create a stereo match algorithm!" << std::endl;
        std::cout << "the resolution of Point cloud Voxel filter : " << resolution_ << std::endl;
        this->resolution = resolution_;
        this->octomapResolution = octomapResolution_ > 0.0 ? octomapResolution_ : resolution_;
        this->removeClass0InDenseMap = removeClass0InDenseMap_;
        voxel.setLeafSize(resolution, resolution, resolution);
        std::cout << "the Point cloud Outlier filter params :   \n"
                  << "meank : " << meank_ << std::endl
                  << "stdthresh :" << stdthresh_ << std::endl;
        this->meank = meank_;
        this->stdthresh = stdthresh_;
        sor.setMeanK(meank);
        sor.setStddevMulThresh(stdthresh);
        std::cout << "the Unit of Point cloud : " << unit_ << std::endl;
        this->unit = unit_;
        std::cout << "the resolution of Semantic OctoMap : " << this->octomapResolution << std::endl;
        std::cout << "remove class-0 points in dense semantic map : " << this->removeClass0InDenseMap << std::endl;
        globalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        semanticGlobalMap.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        stereo = Stereo_Algorithm::create(mindisp_, maxdisp_, type);
        std::cout << "max disp : " << maxdisp_ << " min disp :" << mindisp_ << std::endl;
        numDisp = static_cast<int>(maxdisp_ - mindisp_);
        viewerThread = std::make_unique<std::thread>(std::bind(&PointCloudMapping::viewer, this));
    }

    void PointCloudMapping::shutdown()
    {
        {
            shutDownFlag.store(true);
            keyFrameUpdated.notify_one();
        }
        viewerThread->join();
        save();
    }

    void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth)
    {
        cv::Mat emptyMask;
        insertKeyFrameWithSemanticMask(kf, color, depth, emptyMask);
    }

    void PointCloudMapping::insertKeyFrameWithSemanticMask(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat &semanticMask)
    {
        mSensor = MappingSensor::RGBD;
        cout << "receive a keyframe, id = " << kf->mnId << endl;
        const bool hasValidMask = !semanticMask.empty() && semanticMask.type() == CV_16UC1 &&
                                  semanticMask.rows == depth.rows && semanticMask.cols == depth.cols;
        std::cout << "[SemanticPointCloud] insert KF id=" << kf->mnId
                  << ", mask_empty=" << semanticMask.empty()
                  << ", mask_type=" << (semanticMask.empty() ? -1 : semanticMask.type())
                  << ", mask_size=" << (semanticMask.empty() ? 0 : semanticMask.cols)
                  << "x" << (semanticMask.empty() ? 0 : semanticMask.rows)
                  << ", depth_size=" << depth.cols << "x" << depth.rows
                  << ", valid_mask=" << hasValidMask;
        if (!semanticMask.empty() && semanticMask.type() == CV_16UC1)
        {
            std::cout << ", nonzero=" << cv::countNonZero(semanticMask)
                      << ", class0_pixels=" << countSemanticClassPixels(semanticMask, 0);
        }
        std::cout << std::endl;
        lock_guard<mutex> lck(keyFrameUpdateMutex);
        keyframes.push_back(kf);
        colorImgs.push_back(color.clone());
        depthImgs.push_back(depth.clone());
        semanticMaskImgs.push_back(semanticMask.clone());
        keyFrameUpdated.notify_one();
    }

    void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &Q)
    {
        mSensor = MappingSensor::STEREO;
        cout << "receive a keyframe, id = " << kf->mnId << endl;
        lock_guard<mutex> lck(keyFrameUpdateMutex);
        keyframes.push_back(kf);
        colorImgs.push_back(left.clone());
        rightImgs.push_back(right.clone());
        this->Q = Q.clone();
        keyFrameUpdated.notify_one();
    }

    void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &disp, cv::Mat &Q)
    {
        mSensor = MappingSensor::STEREO;
        cout << "receive a keyframe, id = " << kf->mnId << endl;
        lock_guard<mutex> lck(keyFrameUpdateMutex);
        keyframes.push_back(kf);
        colorImgs.push_back(left.clone());
        rightImgs.push_back(right.clone());
        dispImgs.push_back(disp);
        this->Q = Q.clone();
        keyFrameUpdated.notify_one();
    }

    pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::GetPointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth)
    {
        cv::Mat emptyMask;
        return GetSemanticPointCloud(kf, color, depth, emptyMask, false);
    }

    pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::GetSemanticPointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, cv::Mat &semanticMask, bool useSemanticColor)
    {
        PointCloud::Ptr tmp(new PointCloud());
        if (depth.type() != CV_32F)
            depth.convertTo(depth, CV_32F);

        const bool hasSemanticMask = !semanticMask.empty() && semanticMask.type() == CV_16UC1 &&
                                     semanticMask.rows == depth.rows && semanticMask.cols == depth.cols;
        size_t validDepthPoints = 0;
        size_t skippedClass0Points = 0;

        for (int m = 0; m < depth.rows; m += 1)
        {
            for (int n = 0; n < depth.cols; n += 1)
            {
                const float d = depth.ptr<float>(m)[n];
                if (d / unit < 0.01 || d / unit > 10.0)
                    continue;
                ++validDepthPoints;

                uint16_t pixelValue = 0;
                if (hasSemanticMask)
                {
                    pixelValue = semanticMask.ptr<uint16_t>(m)[n];
                    if (removeClass0InDenseMap && pixelValue > 0 && static_cast<int>(pixelValue / 100) == 0)
                    {
                        ++skippedClass0Points;
                        continue;
                    }
                }

                PointT p;
                p.z = d / unit;
                p.x = (n - kf->cx) * p.z / kf->fx;
                p.y = (m - kf->cy) * p.z / kf->fy;

                p.b = color.ptr<uchar>(m)[n * 3];
                p.g = color.ptr<uchar>(m)[n * 3 + 1];
                p.r = color.ptr<uchar>(m)[n * 3 + 2];

                if (useSemanticColor && pixelValue > 0)
                    colorBySemanticClass(pixelValue, p.r, p.g, p.b);

                tmp->points.push_back(p);
            }
        }

        static size_t semanticPointCloudDebugCount = 0;
        if (hasSemanticMask && (semanticPointCloudDebugCount < 20 || semanticPointCloudDebugCount % 50 == 0))
        {
            std::cout << "[SemanticPointCloud] build KF id=" << kf->mnId
                      << ", useSemanticColor=" << useSemanticColor
                      << ", remove_class0_dense_map=" << removeClass0InDenseMap
                      << ", valid_depth_points=" << validDepthPoints
                      << ", skipped_class0_depth_points=" << skippedClass0Points
                      << ", output_points_before_transform=" << tmp->points.size()
                      << std::endl;
        }
        if (hasSemanticMask)
            ++semanticPointCloudDebugCount;

        PointCloud::Ptr cloud(new PointCloud());
        Eigen::Matrix4f transform = kf->GetPoseInverse().matrix();
        cloud->points.resize(tmp->points.size());

        for (size_t i = 0; i < tmp->points.size(); ++i)
        {
            const PointT &pt = tmp->points[i];
            Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4f vt = transform * v;
            cloud->points[i].x = vt[0];
            cloud->points[i].y = vt[1];
            cloud->points[i].z = vt[2];
            cloud->points[i].r = pt.r;
            cloud->points[i].g = pt.g;
            cloud->points[i].b = pt.b;
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::GetPointCloud(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &Q)
    {
        cv::Mat left_r = left.clone();
        cv::Mat right_r = right.clone();
        cv::Mat disp = stereo->inference(left_r, right_r);
        double min = 0;
        double max = 0;
        cv::minMaxLoc(disp, &min, &max);
        std::cout << "min disp : " << min << " max disp : " << max << std::endl;
        cv::Mat disp_vis = ((disp - min) / (max - min)) * 255;
        disp_vis.convertTo(disp_vis, CV_8U);
        cv::imshow("disp_vis", disp_vis);
        cv::waitKey(1);

        cv::Mat points_image;
        cv::reprojectImageTo3D(disp, points_image, Q);
        points_image.convertTo(points_image, CV_32F);
        PointCloud::Ptr tmp(new PointCloud());

        for (int m = static_cast<int>(points_image.rows * 0); m < points_image.rows; m += 1)
        {
            for (int n = static_cast<int>(points_image.cols * 0); n < points_image.cols; n += 1)
            {
                if ((points_image.ptr<float>(m)[3 * n + 2] / unit) > 50 ||
                    (points_image.ptr<float>(m)[3 * n + 2] / unit) < 1.0)
                    continue;
                PointT p;
                p.x = points_image.ptr<float>(m)[3 * n];
                p.y = points_image.ptr<float>(m)[3 * n + 1];
                p.z = points_image.ptr<float>(m)[3 * n + 2];
                p.b = left_r.ptr<uchar>(m)[n * 3];
                p.g = left_r.ptr<uchar>(m)[n * 3 + 1];
                p.r = left_r.ptr<uchar>(m)[n * 3 + 2];
                tmp->points.push_back(p);
            }
        }
        PointCloud::Ptr cloud(new PointCloud());
        Eigen::Matrix4f transform = kf->GetPoseInverse().matrix();
        cloud->points.resize(tmp->points.size());

        for (size_t i = 0; i < tmp->points.size(); ++i)
        {
            const PointT &pt = tmp->points[i];
            Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4f vt = transform * v;
            cloud->points[i].x = vt[0];
            cloud->points[i].y = vt[1];
            cloud->points[i].z = vt[2];
            cloud->points[i].r = pt.r;
            cloud->points[i].g = pt.g;
            cloud->points[i].b = pt.b;
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        cout << "generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;
        return cloud;
    }

    pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::GetPointCloud(KeyFrame *kf, cv::Mat &left, cv::Mat &right, cv::Mat &disp, cv::Mat &Q)
    {
        cv::Mat disp_c = disp.clone();
        cv::Mat left_r = left.clone();
        if (disp_c.type() != CV_16S)
            disp_c.convertTo(disp_c, CV_16S);
        cv::Mat points_image;
        cv::reprojectImageTo3D(disp_c, points_image, Q, true, CV_32F);
        PointCloud::Ptr tmp(new PointCloud());
        for (int m = static_cast<int>(points_image.rows * 0.25); m < static_cast<int>(points_image.rows * 0.75); m += 1)
        {
            for (int n = static_cast<int>(points_image.cols * 0.25); n < static_cast<int>(points_image.cols * 0.75); n += 1)
            {
                if ((points_image.ptr<float>(m)[3 * n + 2] / unit) > 10 ||
                    (points_image.ptr<float>(m)[3 * n + 2] / unit) < 0.3)
                    continue;
                PointT p;
                p.x = points_image.ptr<float>(m)[3 * n];
                p.y = points_image.ptr<float>(m)[3 * n + 1];
                p.z = points_image.ptr<float>(m)[3 * n + 2];
                p.b = left_r.ptr<uchar>(m)[n * 3];
                p.g = left_r.ptr<uchar>(m)[n * 3 + 1];
                p.r = left_r.ptr<uchar>(m)[n * 3 + 2];
                tmp->points.push_back(p);
            }
        }
        PointCloud::Ptr cloud(new PointCloud());
        Eigen::Matrix4f transform = kf->GetPoseInverse().matrix();
        cloud->points.resize(tmp->points.size());

        for (size_t i = 0; i < tmp->points.size(); ++i)
        {
            const PointT &pt = tmp->points[i];
            Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0);
            Eigen::Vector4f vt = transform * v;
            cloud->points[i].x = vt[0];
            cloud->points[i].y = vt[1];
            cloud->points[i].z = vt[2];
            cloud->points[i].r = pt.r;
            cloud->points[i].g = pt.g;
            cloud->points[i].b = pt.b;
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    void PointCloudMapping::viewer()
    {
        while (1)
        {
            size_t N = 0;
            {
                unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
                keyFrameUpdated.wait(lck_keyframeUpdated, [this]()
                                     { return shutDownFlag.load() || lastKeyframeSize < keyframes.size(); });

                if (shutDownFlag.load())
                {
                    break;
                }
                N = keyframes.size();
            }
            PointCloud::Ptr p;
            if (mSensor == MappingSensor::RGBD)
            {
                for (size_t i = lastKeyframeSize; i < N; i++)
                {
                    const bool hasMask = i < semanticMaskImgs.size() && !semanticMaskImgs[i].empty();
                    if (hasMask)
                    {
                        if (removeClass0InDenseMap)
                        {
                            const size_t removedFromGlobal = removeSemanticClassPointsFromCloud(globalMap, keyframes[i], depthImgs[i], semanticMaskImgs[i], 0, unit);
                            const size_t removedFromSemanticGlobal = removeSemanticClassPointsFromCloud(semanticGlobalMap, keyframes[i], depthImgs[i], semanticMaskImgs[i], 0, unit);
                            if (removedFromGlobal > 0 || removedFromSemanticGlobal > 0)
                            {
                                std::cout << "[SemanticPointCloud] erase existing class0 points by KF id=" << keyframes[i]->mnId
                                          << ", removed_global=" << removedFromGlobal
                                          << ", removed_semantic_global=" << removedFromSemanticGlobal
                                          << std::endl;
                            }
                        }
                        p = GetSemanticPointCloud(keyframes[i], colorImgs[i], depthImgs[i], semanticMaskImgs[i], false);
                    }
                    else
                    {
                        std::cout << "[SemanticPointCloud] KF id=" << keyframes[i]->mnId
                                  << " has no semantic mask, building normal point cloud." << std::endl;
                        p = GetPointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
                    }
                    (*globalMap) += *p;

                    if (hasMask)
                    {
                        PointCloud::Ptr semanticP = GetSemanticPointCloud(keyframes[i], colorImgs[i], depthImgs[i], semanticMaskImgs[i]);
                        (*semanticGlobalMap) += *semanticP;
                        hasSemanticCloud = true;
                    }
                }
            }
            else if (mSensor == MappingSensor::STEREO)
            {
                for (size_t i = lastKeyframeSize; i < N; i++)
                {
                    if (dispImgs.size() == 0)
                    {
                        p = GetPointCloud(keyframes[i], colorImgs[i], rightImgs[i], Q);
                        (*globalMap) += *p;
                    }
                    else
                    {
                        p = GetPointCloud(keyframes[i], colorImgs[i], rightImgs[i], dispImgs[i], Q);
                        (*globalMap) += *p;
                    }
                }
            }

            voxel.setInputCloud(globalMap);
            voxel.filter(*globalMap);

            sor.setInputCloud(globalMap);
            sor.filter(*globalMap);

            if (hasSemanticCloud)
            {
                voxel.setInputCloud(semanticGlobalMap);
                voxel.filter(*semanticGlobalMap);
                sor.setInputCloud(semanticGlobalMap);
                sor.filter(*semanticGlobalMap);
            }
            std::cout << "the num of points : " << globalMap->points.size() << std::endl;
            if (hasSemanticCloud)
                std::cout << "the num of semantic points : " << semanticGlobalMap->points.size() << std::endl;
            lastKeyframeSize = N;
        }
    }

    bool PointCloudMapping::saveSemanticOctoMap(const std::string &filename)
    {
        if (semanticGlobalMap == nullptr || semanticGlobalMap->empty())
        {
            std::cout << "[SemanticOctoMap] skip saving, semantic point cloud is empty." << std::endl;
            return false;
        }

        octomap::ColorOcTree tree(octomapResolution);
        size_t inserted = 0;
        for (const PointT &point : semanticGlobalMap->points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                continue;

            octomap::ColorOcTreeNode *node = tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
            if (node != nullptr)
            {
                node->setColor(point.r, point.g, point.b);
                ++inserted;
            }
        }

        if (inserted == 0)
        {
            std::cout << "[SemanticOctoMap] skip saving, no valid points were inserted." << std::endl;
            return false;
        }

        tree.updateInnerOccupancy();
        const bool savedColorTree = tree.write(filename);

        std::string binaryFilename = filename;
        const size_t dotPos = binaryFilename.find_last_of('.');
        if (dotPos != std::string::npos)
            binaryFilename = binaryFilename.substr(0, dotPos);
        binaryFilename += ".bt";
        const bool savedBinaryTree = tree.writeBinary(binaryFilename);

        std::cout << "[SemanticOctoMap] inserted_points=" << inserted
                  << ", occupied_leaf_nodes=" << tree.getNumLeafNodes()
                  << ", resolution=" << octomapResolution
                  << ", saved_color_tree=" << filename << "(" << savedColorTree << ")"
                  << ", saved_binary_tree=" << binaryFilename << "(" << savedBinaryTree << ")"
                  << std::endl;

        return savedColorTree;
    }

    void PointCloudMapping::save()
    {
        if (this->globalMap != nullptr)
            pcl::io::savePLYFile("/home/wen/orbslam_dense/src/orbslam3_dense_ros_rgbd/ORBSLAM3_Dense-main/PointCloudmapping.ply", *globalMap);
        if (this->semanticGlobalMap != nullptr && hasSemanticCloud)
        {
            pcl::io::savePLYFile("/home/wen/orbslam_dense/src/orbslam3_dense_ros_rgbd/ORBSLAM3_Dense-main/SemanticPointCloudmapping.ply", *semanticGlobalMap);
            saveSemanticOctoMap("/home/wen/orbslam_dense/src/orbslam3_dense_ros_rgbd/ORBSLAM3_Dense-main/SemanticOctoMap.ot");
        }
    }
}
