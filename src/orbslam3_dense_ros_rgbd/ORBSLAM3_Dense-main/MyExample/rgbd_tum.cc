#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>

static volatile std::sig_atomic_t gRequestShutdown = 0;

static void SignalHandler(int)
{
    gRequestShutdown = 1;
}

struct ImageEntry
{
    double timestamp;
    std::string path;
};

struct RGBDPair
{
    double timestamp;
    std::string rgbPath;
    std::string depthPath;
};

static bool LoadImageList(const std::string &filename, std::vector<ImageEntry> &entries)
{
    entries.clear();

    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
        std::cerr << "Cannot open association file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream iss(line);
        ImageEntry entry;
        if (!(iss >> entry.timestamp >> entry.path))
            continue;

        entries.push_back(entry);
    }

    std::sort(entries.begin(), entries.end(),
              [](const ImageEntry &a, const ImageEntry &b) { return a.timestamp < b.timestamp; });
    return !entries.empty();
}

static bool TryParseTimestampFromPath(const std::string &imagePath, double &timestamp)
{
    const std::string stem = boost::filesystem::path(imagePath).stem().string();
    try
    {
        size_t parsedChars = 0;
        timestamp = std::stod(stem, &parsedChars);
        return parsedChars == stem.size();
    }
    catch (const std::exception &)
    {
        return false;
    }
}

static bool TryUseFilenameTimestamps(std::vector<ImageEntry> &entries)
{
    if (entries.empty())
        return false;

    std::vector<double> parsedTimestamps(entries.size(), 0.0);
    size_t parsedCount = 0;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (TryParseTimestampFromPath(entries[i].path, parsedTimestamps[i]))
            ++parsedCount;
    }

    if (parsedCount * 2 < entries.size())
        return false;

    for (size_t i = 0; i < entries.size(); ++i)
        entries[i].timestamp = parsedTimestamps[i];

    std::sort(entries.begin(), entries.end(),
              [](const ImageEntry &a, const ImageEntry &b) { return a.timestamp < b.timestamp; });
    return true;
}

static bool TimestampRangesLookCompatible(const std::vector<ImageEntry> &a, const std::vector<ImageEntry> &b)
{
    if (a.empty() || b.empty())
        return false;

    const double firstDiff = std::abs(a.front().timestamp - b.front().timestamp);
    const double lastDiff = std::abs(a.back().timestamp - b.back().timestamp);
    return firstDiff < 10.0 && lastDiff < 10.0;
}

static std::vector<RGBDPair> AssociateRGBD(const std::vector<ImageEntry> &rgbEntries,
                                           const std::vector<ImageEntry> &depthEntries,
                                           double maxTimeDiff)
{
    std::vector<RGBDPair> pairs;
    size_t depthIndex = 0;

    for (const ImageEntry &rgb : rgbEntries)
    {
        while (depthIndex + 1 < depthEntries.size() &&
               std::abs(depthEntries[depthIndex + 1].timestamp - rgb.timestamp) <
                   std::abs(depthEntries[depthIndex].timestamp - rgb.timestamp))
        {
            ++depthIndex;
        }

        if (depthIndex < depthEntries.size() &&
            std::abs(depthEntries[depthIndex].timestamp - rgb.timestamp) <= maxTimeDiff)
        {
            RGBDPair pair;
            pair.timestamp = rgb.timestamp;
            pair.rgbPath = rgb.path;
            pair.depthPath = depthEntries[depthIndex].path;
            pairs.push_back(pair);
        }
    }

    return pairs;
}

static bool FindNearestImage(const std::vector<ImageEntry> &entries,
                             double timestamp,
                             double maxTimeDiff,
                             size_t &index,
                             ImageEntry &matched,
                             double &timeDiff)
{
    if (entries.empty())
        return false;

    while (index + 1 < entries.size() &&
           std::abs(entries[index + 1].timestamp - timestamp) <
               std::abs(entries[index].timestamp - timestamp))
    {
        ++index;
    }

    timeDiff = std::abs(entries[index].timestamp - timestamp);
    if (timeDiff > maxTimeDiff)
        return false;

    matched = entries[index];
    return true;
}

static std::string ResolveDatasetPath(const boost::filesystem::path &root, const std::string &relativePath)
{
    boost::filesystem::path path(relativePath);
    if (path.is_absolute())
        return path.string();
    return (root / path).string();
}

static void SleepInterruptible(double seconds)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (!gRequestShutdown && std::chrono::steady_clock::now() < deadline)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            break;
        const auto remaining = deadline - now;
        if (remaining > std::chrono::milliseconds(20))
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        else
            std::this_thread::sleep_for(remaining);
    }
}

int main(int argc, char **argv)
{
    std::string rootDir;
    std::string settingsPath;
    std::string vocabularyPath;
    std::string rgbFile;
    std::string depthFile;
    std::string semanticFile;
    std::string semanticDir;
    std::string trajectoryFile;
    std::string keyframeTrajectoryFile;
    bool useViewer = true;
    bool realtime = false;
    bool enableSemantic = true;
    bool semanticDebug = true;
    bool preferSemanticNameMatch = true;
    double maxTimeDiff = 0.02;
    double maxSemanticTimeDiff = 0.02;
    int semanticDebugFrames = 10;
    int semanticDebugInterval = 50;

    boost::program_options::options_description desc("TUM RGB-D dataset runner");
    desc.add_options()
        ("help,h", "show help message")
        ("root_dir,r", boost::program_options::value<std::string>(&rootDir)->required(), "TUM sequence root directory")
        ("param_path,p", boost::program_options::value<std::string>(&settingsPath)->default_value("MyExample/rgbd_orb.yaml"), "camera settings yaml")
        ("voc_path,v", boost::program_options::value<std::string>(&vocabularyPath)->default_value("Vocabulary/ORBvoc.txt"), "ORB vocabulary path")
        ("rgb_file", boost::program_options::value<std::string>(&rgbFile)->default_value("rgb.txt"), "RGB timestamp file")
        ("depth_file", boost::program_options::value<std::string>(&depthFile)->default_value("depth.txt"), "depth timestamp file")
        ("semantic_file", boost::program_options::value<std::string>(&semanticFile)->default_value("semantic.txt"), "semantic mask timestamp file")
        ("semantic_dir", boost::program_options::value<std::string>(&semanticDir)->default_value("semantic"), "semantic mask directory for RGB-name matching")
        ("max_time_diff", boost::program_options::value<double>(&maxTimeDiff)->default_value(0.02), "maximum RGB/depth timestamp difference in seconds")
        ("max_semantic_time_diff", boost::program_options::value<double>(&maxSemanticTimeDiff)->default_value(0.02), "maximum RGB/semantic timestamp difference in seconds")
        ("trajectory", boost::program_options::value<std::string>(&trajectoryFile)->default_value("CameraTrajectory.txt"), "output camera trajectory path")
        ("keyframe_trajectory", boost::program_options::value<std::string>(&keyframeTrajectoryFile)->default_value("KeyFrameTrajectory.txt"), "output keyframe trajectory path")
        ("enable_semantic", boost::program_options::value<bool>(&enableSemantic)->default_value(true), "load semantic masks from semantic_file")
        ("prefer_semantic_name_match", boost::program_options::value<bool>(&preferSemanticNameMatch)->default_value(true), "prefer semantic_dir/RGB_filename.png over timestamp matching")
        ("semantic_debug", boost::program_options::value<bool>(&semanticDebug)->default_value(true), "print semantic loading debug information")
        ("semantic_debug_frames", boost::program_options::value<int>(&semanticDebugFrames)->default_value(10), "print semantic debug for the first N frames")
        ("semantic_debug_interval", boost::program_options::value<int>(&semanticDebugInterval)->default_value(50), "print semantic debug every N frames after the first frames")
        ("viewer", boost::program_options::value<bool>(&useViewer)->default_value(true), "enable Pangolin viewer")
        ("realtime", boost::program_options::value<bool>(&realtime)->default_value(false), "sleep according to dataset timestamps");

    boost::program_options::variables_map vm;
    try
    {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        boost::program_options::notify(vm);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    const boost::filesystem::path rootPath(rootDir);

    std::vector<ImageEntry> rgbEntries;
    std::vector<ImageEntry> depthEntries;
    std::vector<ImageEntry> semanticEntries;
    if (!LoadImageList(ResolveDatasetPath(rootPath, rgbFile), rgbEntries) ||
        !LoadImageList(ResolveDatasetPath(rootPath, depthFile), depthEntries))
    {
        return 1;
    }

    bool useSemantic = false;
    std::string semanticFilePath;
    if (enableSemantic && !semanticFile.empty())
    {
        semanticFilePath = ResolveDatasetPath(rootPath, semanticFile);
        std::cout << "[SemanticDebug] enable_semantic=true, semantic_file=" << semanticFilePath << std::endl;
        if (preferSemanticNameMatch)
            std::cout << "[SemanticDebug] prefer RGB-name semantic matching, semantic_dir="
                      << ResolveDatasetPath(rootPath, semanticDir) << std::endl;
        if (boost::filesystem::exists(semanticFilePath))
        {
            useSemantic = LoadImageList(semanticFilePath, semanticEntries);
            if (!useSemantic)
                std::cerr << "Semantic file exists but no entries were loaded: " << semanticFilePath << std::endl;
            else if (!TimestampRangesLookCompatible(rgbEntries, semanticEntries))
            {
                std::cout << "[SemanticDebug] semantic timestamps do not match RGB timestamp range. "
                          << "Trying to recover timestamps from semantic mask filenames." << std::endl;
                if (TryUseFilenameTimestamps(semanticEntries))
                    std::cout << "[SemanticDebug] recovered semantic timestamps from filenames." << std::endl;
                else
                    std::cout << "[SemanticDebug] failed to recover semantic timestamps from filenames." << std::endl;
            }
        }
        else
        {
            std::cout << "Semantic file not found, running RGB-D only: " << semanticFilePath << std::endl;
        }
    }
    else
    {
        std::cout << "[SemanticDebug] semantic loading disabled by arguments." << std::endl;
    }

    const std::vector<RGBDPair> pairs = AssociateRGBD(rgbEntries, depthEntries, maxTimeDiff);
    if (pairs.empty())
    {
        std::cerr << "No RGB-D pairs found. Try increasing --max_time_diff." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << rgbEntries.size() << " RGB entries, "
              << depthEntries.size() << " depth entries, associated "
              << pairs.size() << " RGB-D pairs." << std::endl;
    if (useSemantic)
    {
        std::cout << "Loaded " << semanticEntries.size() << " semantic entries." << std::endl;
        if (semanticDebug && !semanticEntries.empty())
        {
            std::cout << "[SemanticDebug] first semantic entry: t=" << semanticEntries.front().timestamp
                      << ", path=" << semanticEntries.front().path << std::endl;
            std::cout << "[SemanticDebug] last semantic entry: t=" << semanticEntries.back().timestamp
                      << ", path=" << semanticEntries.back().path << std::endl;
        }
    }

    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::RGBD, useViewer);

    size_t semanticIndex = 0;
    size_t usedSemanticFrames = 0;
    size_t semanticMatchedFrames = 0;
    size_t semanticNoMatchFrames = 0;
    size_t semanticLoadFailedFrames = 0;
    size_t semanticWrongTypeFrames = 0;
    size_t semanticSizeMismatchFrames = 0;
    for (size_t i = 0; i < pairs.size(); ++i)
    {
        if (gRequestShutdown)
        {
            std::cout << "\n[Signal] Ctrl+C received, stopping dataset playback and saving map..." << std::endl;
            break;
        }

        const std::string rgbPath = ResolveDatasetPath(rootPath, pairs[i].rgbPath);
        const std::string depthPath = ResolveDatasetPath(rootPath, pairs[i].depthPath);

        cv::Mat rgb = cv::imread(rgbPath, cv::IMREAD_UNCHANGED);
        cv::Mat depth = cv::imread(depthPath, cv::IMREAD_UNCHANGED);

        if (rgb.empty())
        {
            std::cerr << "Failed to load RGB image: " << rgbPath << std::endl;
            continue;
        }
        if (depth.empty())
        {
            std::cerr << "Failed to load depth image: " << depthPath << std::endl;
            continue;
        }

        cv::Mat semanticMask;
        if (useSemantic)
        {
            ImageEntry semanticEntry;
            double semanticTimeDiff = -1.0;
            std::string semanticPath;
            std::string semanticMatchMode = "none";
            const bool shouldPrintSemanticDebug = semanticDebug &&
                                                  (static_cast<int>(i) < semanticDebugFrames ||
                                                   (semanticDebugInterval > 0 && i % static_cast<size_t>(semanticDebugInterval) == 0));
            if (preferSemanticNameMatch)
            {
                const boost::filesystem::path rgbRelPath(pairs[i].rgbPath);
                const std::string semanticName = rgbRelPath.stem().string() + ".png";
                semanticPath = ResolveDatasetPath(rootPath, (boost::filesystem::path(semanticDir) / semanticName).string());
                if (boost::filesystem::exists(semanticPath))
                {
                    semanticMatchMode = "rgb_name";
                    semanticTimeDiff = 0.0;
                }
                else
                {
                    semanticPath.clear();
                }
            }

            if (semanticPath.empty() &&
                FindNearestImage(semanticEntries, pairs[i].timestamp, maxSemanticTimeDiff, semanticIndex, semanticEntry, semanticTimeDiff))
            {
                semanticPath = ResolveDatasetPath(rootPath, semanticEntry.path);
                semanticMatchMode = "timestamp";
            }

            if (!semanticPath.empty())
            {
                ++semanticMatchedFrames;
                semanticMask = cv::imread(semanticPath, cv::IMREAD_UNCHANGED);
                if (semanticMask.empty())
                {
                    ++semanticLoadFailedFrames;
                    std::cerr << "Failed to load semantic mask: " << semanticPath << std::endl;
                }
                else if (semanticMask.type() != CV_16UC1)
                {
                    ++semanticWrongTypeFrames;
                    std::cerr << "Semantic mask must be 16-bit single channel, got type "
                              << semanticMask.type() << ": " << semanticPath << std::endl;
                    semanticMask.release();
                }
                else if (semanticMask.size() != rgb.size())
                {
                    ++semanticSizeMismatchFrames;
                    std::cerr << "Semantic mask size mismatch: mask " << semanticMask.cols << "x" << semanticMask.rows
                              << ", rgb " << rgb.cols << "x" << rgb.rows << ": " << semanticPath << std::endl;
                    semanticMask.release();
                }
                else
                {
                    ++usedSemanticFrames;
                }

                if (shouldPrintSemanticDebug)
                {
                    const int nonZeroPixels = semanticMask.empty() ? 0 : cv::countNonZero(semanticMask);
                    std::cout << "[SemanticDebug] frame=" << i
                              << ", rgb_t=" << pairs[i].timestamp
                              << ", semantic_t=" << (semanticMatchMode == "timestamp" ? semanticEntry.timestamp : pairs[i].timestamp)
                              << ", dt=" << semanticTimeDiff
                              << ", match_mode=" << semanticMatchMode
                              << ", semantic_path=" << semanticPath
                              << ", loaded=" << (!semanticMask.empty())
                              << ", type=" << (semanticMask.empty() ? -1 : semanticMask.type())
                              << ", size=" << (semanticMask.empty() ? 0 : semanticMask.cols)
                              << "x" << (semanticMask.empty() ? 0 : semanticMask.rows)
                              << ", nonzero=" << nonZeroPixels
                              << std::endl;
                }
            }
            else
            {
                ++semanticNoMatchFrames;
                if (shouldPrintSemanticDebug)
                {
                    std::cout << "[SemanticDebug] frame=" << i
                              << ", rgb_t=" << pairs[i].timestamp
                              << ", no semantic match within " << maxSemanticTimeDiff << " s"
                              << std::endl;
                }
            }
        }

        if (!semanticMask.empty())
            SLAM.TrackRGBD(rgb, depth, semanticMask, pairs[i].timestamp, std::vector<ORB_SLAM3::IMU::Point>(), rgbPath);
        else
            SLAM.TrackRGBD(rgb, depth, pairs[i].timestamp, std::vector<ORB_SLAM3::IMU::Point>(), rgbPath);

        if (realtime && i + 1 < pairs.size())
        {
            const double waitSeconds = pairs[i + 1].timestamp - pairs[i].timestamp;
            if (waitSeconds > 0.0)
                SleepInterruptible(waitSeconds);
        }
    }

    std::cout << "Shutting down SLAM and saving map..." << std::endl;
    SLAM.Shutdown();
    if (useSemantic)
    {
        std::cout << "Used semantic masks for " << usedSemanticFrames << " frames." << std::endl;
        std::cout << "[SemanticDebug] summary: matched=" << semanticMatchedFrames
                  << ", used=" << usedSemanticFrames
                  << ", no_match=" << semanticNoMatchFrames
                  << ", load_failed=" << semanticLoadFailedFrames
                  << ", wrong_type=" << semanticWrongTypeFrames
                  << ", size_mismatch=" << semanticSizeMismatchFrames
                  << std::endl;
    }

    if (!trajectoryFile.empty())
        SLAM.SaveTrajectoryTUM(trajectoryFile);
    if (!keyframeTrajectoryFile.empty())
        SLAM.SaveKeyFrameTrajectoryTUM(keyframeTrajectoryFile);

    return 0;
}
