EXE := test
CC := g++ -std=c++11 -O3
#CC := g++ -std=c++11 -g
#CC := g++ 

DEFINES :=
#DEFINES := -DMY_VIS_MODE

CC_FLAG :=-I/export/opencv320/include \
		-I/usr/local/include -I/export/home/xxx/cuda-7.5/include/ -I/usr/include/ \
		-I/export/home/xxx/caffe-ssd/include/ \
		-I/export/home/xxx/caffe-ssd/build/src/ \
		-I./include

CC_FLAG += $(DEFINES)

LD_FLAG :=-L/export/opencv320/lib/ \
		-L/usr/lib64/ \
		-lopencv_core \
		-lopencv_highgui \
		-lopencv_imgproc \
		-lopencv_features2d \
		-lopencv_flann \
		-lopencv_video \
		-lopencv_videoio \
		-lopencv_imgcodecs \
		-lopencv_calib3d \
		-lopencv_tracking


LD_FLAG += -L/usr/lib64/ \
		-lglog \
		-lboost_system \
		-lgflags

LD_FLAG += -L/export/home/xxx/caffe-ssd/build/lib/ \
		-lcaffe

$(EXE): obj/utils.o obj/detector.o obj/perspective.o obj/main.o
	$(CC) $(LD_FLAG) $^ -o $@


obj/utils.o : src/utils.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

obj/detector.o : src/detector.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@


obj/perspective.o : src/perspective.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

obj/main.o : src/main.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

clean:
	rm -r obj/*
	rm test
