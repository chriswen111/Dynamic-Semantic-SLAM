# Dynamic-Semantic-SLAM

基于 ORB-SLAM3 的 RGB-D SLAM 项目，包含稠密点云重建、外部语义掩码输入和语义 OctoMap 保存功能。提供离线 TUM RGB-D 数据集入口及 ROS 1 RGB-D 节点。

> 本说明根据当前源码整理，尚未在新的 Linux 环境中完成编译和运行验证。当前仓库不包含数据集、编译产物和语义分割模型；使用语义功能时，需要自行生成或发布符合下述格式的掩码。

## 1. 环境依赖

项目面向 Linux，ROS 接口使用 ROS 1/catkin。源码中的工作空间路径指向 ROS Noetic，但这不代表所有依赖版本均已验证。

- CMake 3.15 或以上，以及支持 C++14 的 C++ 编译器。
- OpenCV、Eigen3、Pangolin、PCL、OctoMap。
- Boost filesystem、program_options、serialization，以及 OpenSSL 开发库。
- 仓库已包含 DBoW2、g2o、ELAS 和 Sophus 源码。
- ROS 模式额外需要 roscpp、roslib、sensor_msgs、cv_bridge、image_transport、message_filters。

安装各依赖时，应让 CMake 能找到其配置文件；使用自定义安装目录时，可设置 `CMAKE_PREFIX_PATH`，或相应的 `OpenCV_DIR`、`Pangolin_DIR` 等变量。ROS 模式下还应检查 OpenCV 与 cv_bridge 的版本兼容性。

## 2. 下载源码和词典

```bash
git clone https://github.com/chriswen111/Dynamic-Semantic-SLAM.git
cd Dynamic-Semantic-SLAM
export SLAM_DIR="$PWD/src/orbslam3_dense_ros_rgbd/ORBSLAM3_Dense-main"
cd "$SLAM_DIR"
tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
test -f Vocabulary/ORBvoc.txt
```

后续命令使用同一终端中的 `SLAM_DIR` 变量。新开终端时，需要重新设置为项目实际路径。


## 3. 编译离线数据集程序

完成依赖安装及上面的 Orbbec 处理后，可以不使用 catkin 编译：

```bash
cd "$SLAM_DIR"
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_DISABLE_FIND_PACKAGE_catkin=TRUE
cmake --build build --target rgbd_tum -j2
./MyExample/rgbd_tum --help
```

该模式下，`rgbd_tum` 可执行文件输出到 `MyExample/`。`-j2` 可按可用内存调整。DBoW2、g2o 和 ELAS 由主项目的 CMake 配置一并构建。

## 4. 使用 TUM RGB-D 数据集

自行准备数据集并解压，目录中至少应包含：

```text
sequence/
├── rgb/
├── depth/
├── rgb.txt
└── depth.txt
```

`rgb.txt` 和 `depth.txt` 每行使用“时间戳 相对图像路径”格式，程序会按时间戳自动关联 RGB 与深度图，默认最大时间差为 0.02 秒。

先复制相机配置模板：

```bash
cd "$SLAM_DIR"
cp MyExample/rgbd_orb.yaml MyExample/my_camera.yaml
```

根据数据集修改 `my_camera.yaml` 中的相机内参、畸变、分辨率、帧率、`Camera.RGB` 和 `RGBD.DepthMapFactor`。不要直接把示例参数用于所有相机或数据集。仓库的 `TUM1.yaml` 可用于参考，但其中的标定参数不能直接套用到 Freiburg3 序列；若使用该文件进行稠密重建，还需参考 `rgbd_orb.yaml` 补充 `PointCloudMapping.*` 配置。

深度缩放因子应与输入数据一致：深度整数除以该因子后得到米。例如输入以毫米保存时通常使用 1000，标准 TUM 深度图通常使用 5000，输入已为浮点米值时使用 1。稠密重建使用的 `PointCloudMapping.Unit` 也应结合实际输入及源码的深度处理核对。

先关闭语义功能，检查普通 RGB-D 流程：

```bash
./MyExample/rgbd_tum \
  --root_dir /absolute/path/to/sequence \
  --param_path MyExample/my_camera.yaml \
  --voc_path Vocabulary/ORBvoc.txt \
  --enable_semantic false \
  --trajectory CameraTrajectory.txt \
  --keyframe_trajectory KeyFrameTrajectory.txt
```

将 `/absolute/path/to/sequence` 替换为实际数据集目录。可以添加 `--realtime true` 按数据集时间间隔播放。

## 5. 输入语义掩码

语义掩码必须是与 RGB 图像同尺寸的 **16 位单通道图像（CV_16UC1）**。当前代码按 `pixelValue / 100` 的整数结果解析类别，像素值 0 不参与该类别判断；普通 8 位类别图或彩色可视化图不能直接使用。

在数据集目录中准备 `semantic/` 和 `semantic.txt`。`semantic.txt` 每行格式为：

```text
1341845822.925900 semantic/1341845822.925900.png
```

默认优先查找 `semantic/` 下与 RGB 图像同名的掩码，再尝试按 `semantic.txt` 时间戳关联。准备好文件后运行：

```bash
./MyExample/rgbd_tum \
  --root_dir /absolute/path/to/sequence \
  --param_path MyExample/my_camera.yaml \
  --voc_path Vocabulary/ORBvoc.txt \
  --enable_semantic true \
  --semantic_file semantic.txt \
  --semantic_dir semantic
```

检查控制台中的 `[SemanticDebug]` 日志和结束时的 `used` 计数，确认掩码实际被读取。动态类别的编号约定需要与 `src/Tracking.cc` 中的过滤逻辑一致。

## 6. ROS 1 实时输入

先补齐第 3 节说明的 catkin 配置，将主项目作为 `orbslam3_dense` 包放入有效工作空间，编译并加载该工作空间的 `devel/setup.bash`。确认能通过 `rospack find orbslam3_dense` 找到主项目。

启动 ROS master、相机驱动或 rosbag 数据源，确保 RGB 与深度图已对齐，且相机配置与输入一致。随后运行：

```bash
roslaunch orbslam3_dense rgbd_ros2.launch \
  voc_file:="$SLAM_DIR/Vocabulary/ORBvoc.txt" \
  settings_file:="$SLAM_DIR/MyExample/my_camera.yaml" \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  enable_semantic_mask:=false
```

`rgbd_ros2.launch` 虽然名称带有 `2`，仍是 ROS 1 launch 文件。这里显式覆盖了词典路径，避免使用文件中写死的原作者路径。

接入外部语义节点时，将 `enable_semantic_mask` 改为 `true`，并设置 `semantic_mask_topic:=/semantic_mask`。掩码消息应为 `mono16` 或 `16UC1`，尺寸与 RGB 一致。默认同步容差 `sync_slop` 为 0.03 秒；启用语义后，需要 RGB、深度和掩码三路消息才能触发处理。

## 7. 结果保存与常见问题

- 离线程序结束后保存 `CameraTrajectory.txt` 和 `KeyFrameTrajectory.txt`，相对路径以运行时当前目录为准。
- 点云和语义地图由 `PointCloudMapping.cc` 保存，包括 `PointCloudmapping.ply`、`SemanticPointCloudmapping.ply` 和 `SemanticOctoMap.ot`，并生成对应的 `.bt` 文件；语义结果需要有效语义点云。运行前必须修改其中的绝对输出路径。
- 使用 `Ctrl+C` 正常退出并等待保存完成，避免直接强制结束进程。
- `ORBvoc.txt` 找不到：检查词典解压位置和传入路径。
- ROS 没有处理图像：检查话题、时间戳、同步容差，以及启用语义后掩码话题是否持续发布。
- `Semantic mask must be 16-bit single channel`：使用 16 位单通道掩码，并保留原始标签编码。
- CMake 或链接失败：检查依赖路径和 Orbbec SDK 专用段落。
- 点云尺度异常：检查相机标定、深度缩放因子及稠密重建参数。

数据集和本地生成的 `build/`、`devel/` 不纳入仓库。后续提交时，应继续排除下载数据和大体积生成文件。
