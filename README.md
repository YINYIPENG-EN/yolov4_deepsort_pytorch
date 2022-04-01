# yolov4_deepsort_pytorch

CSDN使用说明：https://blog.csdn.net/z240626191s/article/details/123892732
权重下载：

百度云权重：

链接：https://pan.baidu.com/s/1SsRx_bmA2aLKGv9OcIRqSw 

提取码：yypn

将下载的目标检测yolov4权重放在yolov4/model_data/文件下

然后执行：python track.py  --source demo.mp4 --classes 0 --show-vid --save-vid
【默认网络是osnet_x0_25，输入上述代码会自动下载权重，权重保存在：C:user/你的用户名/.cache/torch/checkpoints/】

命令说明：      --source 后面的参数是打开摄像头还是本地视频，如果是本地视频输入路径即可

                --classes 是需要检测的类，因为我这里yolo用的是coco数据集，有80个类，如果不加任何参数，默认是检测80个类，如果想只检测某个类，可以单独指定【0指的是指检测人这个类】

                --show-vid则显示图像

                --save-vid是保存跟踪后的视频，保存路径runs/track/exp【每次检测都会生成一个新的exp文件】


如果是自己下载跟踪权重或自己的预权重按下面操作，否则可以忽略：

如果是自己下载的跟踪权重【或者自己训练的】，需要放在deep_sort/deep/checkpoint/文件下，然后根据自己需要的跟踪网络去deep_sort/deep/reid/torchreid/models下修改权重路径。
