from dense_slam_benchmark.dataset_tools.utils import depth2color
import cv2


depth = cv2.imread("/media/spiderman/zhipeng_8t1/datasets/BotanicGarden/1018-00/1018_00_img10hz600p/FoundationStereo/depth/1666059838350278378.tiff", cv2.IMREAD_UNCHANGED)

vis_depth = depth2color(depth, min_depth=0, max_depth=30)
cv2.imshow("1666059836352499962.png", vis_depth)
cv2.waitKey(0)