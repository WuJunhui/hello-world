EXE := test
CC := g++ -std=c++11 -O3
#CC := g++ -std=c++11 -g
#CC := g++ 

DEFINES :=
#DEFINES := -DMY_VIS_MODE

#CC_FLAG := -I/export/home/peiming/book_project_final/book_project/opencv2.4.13_install.debug/include
CC_FLAG :=-I/export/opencv320/include \
		-I/usr/local/include -I/export/home/wjh/cuda-7.5/include/ -I/usr/include/ \
		-I/export/home/wjh/caffe-ssd/include/ \
		-I/export/home/wjh/caffe-ssd/build/src/ \
		-I./include

CC_FLAG += $(DEFINES)

#LD_FLAG := -L/export/home/peiming/book_project_final/book_project/opencv2.4.13_install.debug/lib/
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

LD_FLAG += -L/export/home/wjh/caffe-ssd/build/lib/ \
		-lcaffe

$(EXE): obj/utils.o obj/detector.o obj/heat.o
	$(CC) $(LD_FLAG) $^ -o $@


obj/utils.o : src/utils.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

obj/detector.o : src/detector.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

#obj/extractor.o : src/extractor.cpp
#	$(CC) $(CC_FLAG) -c $^ -o $@

#obj/tracker.o : src/tracker.cpp include/tracker.h
#	$(CC) $(CC_FLAG) -c $< -o $@

obj/heat.o : src/heat.cpp
	$(CC) $(CC_FLAG) -c $^ -o $@

clean:
	rm -r obj/*
	rm test
