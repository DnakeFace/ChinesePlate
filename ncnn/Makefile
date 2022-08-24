
CC = gcc
CPP = g++
STRIP = strip
AR = ar

FLAGS := -Wall -Wno-sign-compare -Wno-unused-result -Wno-unused-function -fPIC -O2
FLAGS += -I./ncnn

LIBS += -lpthread -lrt -ldl ./libs/libncnn.a

SUBDIR = .

INC = $(foreach dir,$(SUBDIR),-I$(dir))
CFLAGS += $(INC) $(FLAGS) -fopenmp
CXXFLAGS += $(INC) -std=c++11 $(FLAGS) -fopenmp

# target source
CSRCS = $(foreach dir,$(SUBDIR),$(wildcard $(dir)/*.c))
CPPSRCS = $(foreach dir,$(SUBDIR),$(wildcard $(dir)/*.cpp))
OBJS += $(CSRCS:%.c=%.o) $(CPPSRCS:%.cpp=%.o)

TARGET := plate
.PHONY : clean all

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CPP) $(CXXFLAGS) -o $@ $^ `pkg-config --cflags --libs opencv` $(LIBS)
	$(STRIP) $(TARGET)

.SUFFIXES: .o .c .cpp .cc .cxx .C

.cpp.o:
	$(CPP) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cc.o:
	$(CPP) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.cxx.o:
	$(CPP) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.C.o:
	$(CPP) -c $(CXXFLAGS) $(INCPATH) -o "$@" "$<"

.c.o:
	$(CC) -c $(CFLAGS) $(INCPATH) -o "$@" "$<"

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
