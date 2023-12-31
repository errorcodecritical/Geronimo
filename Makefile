LIB := ./lib
TEST := ./test
BUILD := ./build
EXE := run

CC := g++
CFLAGS := -larmadillo

API_FILENAMES := graph_sequential layer_relu layer_leaky layer_sigmoid
TEST_FILENAMES := example_xor

API_FILES := $(addsuffix .cpp, $(addprefix $(LIB)/, $(API_FILENAMES)))
TEST_FILES := $(addsuffix .cpp, $(addprefix $(TEST)/, $(TEST_FILENAMES)))

API_OBJS := $(patsubst $(LIB)/%.cpp, $(BUILD)/%.api.o, $(API_FILES)) 
TEST_OBJS := $(patsubst $(TEST)/%.cpp, $(BUILD)/%.test.o, $(TEST_FILES))

all: $(EXE)

$(EXE) : $(API_OBJS) $(TEST_OBJS)
	$(CC) -o $@ $^ $(CFLAGS)

$(BUILD)/%.api.o : $(LIB)/%.cpp
	$(CC) -c -o $@ $^ -I $(LIB)

$(BUILD)/%.test.o : $(TEST)/%.cpp
	$(CC) -c -o $@ $^ -I $(LIB)

clean:
	rm -f $(API_OBJS) $(TEST_OBJS) $(EXE)

.PHONY: all clean
